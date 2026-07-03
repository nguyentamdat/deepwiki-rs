//! Codex OAuth Responses API client.
//!
//! This provider reuses the Codex CLI/IDE ChatGPT OAuth session stored in
//! `$CODEX_HOME/auth.json` or `~/.codex/auth.json`. The file contains bearer
//! tokens and must be treated as secret material.

use anyhow::{Context, Result};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use regex::Regex;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{marker::PhantomData, path::PathBuf, sync::LazyLock};

static JSON_CODE_BLOCK_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```").unwrap());

#[derive(Clone)]
pub struct CodexOAuthClient {
    base_url: String,
    auth: CodexOAuthAuth,
    http: reqwest::Client,
    timeout: std::time::Duration,
}

#[derive(Clone)]
struct CodexOAuthAuth {
    access_token: String,
    account_id: Option<String>,
    is_fedramp: bool,
}

#[derive(Deserialize)]
struct AuthDotJson {
    tokens: Option<TokenData>,
}

#[derive(Deserialize)]
struct TokenData {
    access_token: String,
    #[serde(default)]
    account_id: Option<String>,
    id_token: String,
}

#[derive(Deserialize)]
struct IdClaims {
    #[serde(rename = "https://api.openai.com/auth", default)]
    auth: Option<AuthClaims>,
}

#[derive(Deserialize)]
struct AuthClaims {
    #[serde(default)]
    chatgpt_account_id: Option<String>,
    #[serde(default)]
    chatgpt_account_is_fedramp: bool,
}

impl CodexOAuthClient {
    pub fn new(base_url: &str, timeout_seconds: u64) -> Result<Self> {
        Ok(Self {
            base_url: normalize_codex_base_url(base_url),
            auth: CodexOAuthAuth::from_codex_home()?,
            http: reqwest::Client::new(),
            timeout: std::time::Duration::from_secs(timeout_seconds),
        })
    }

    pub async fn prompt(&self, model: &str, instructions: &str, prompt: &str) -> Result<String> {
        self.responses_text(model, instructions, prompt, None).await
    }

    pub fn extractor<T>(
        &self,
        model: String,
        instructions: String,
        max_retries: u32,
    ) -> CodexOAuthExtractor<T>
    where
        T: JsonSchema + Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
    {
        CodexOAuthExtractor {
            client: self.clone(),
            model,
            instructions,
            max_retries,
            _phantom: PhantomData,
        }
    }

    async fn responses_text(
        &self,
        model: &str,
        instructions: &str,
        prompt: &str,
        text: Option<Value>,
    ) -> Result<String> {
        let mut body = serde_json::json!({
            "model": model,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": prompt
                }]
            }],
            "tool_choice": "none",
            "parallel_tool_calls": false,
            "store": false,
            "stream": true,
            "include": []
        });

        if !instructions.is_empty() {
            body["instructions"] = Value::String(instructions.to_string());
        }
        if let Some(text) = text {
            body["text"] = text;
        }

        let response = self
            .http
            .post(format!("{}/responses", self.base_url))
            .headers(self.auth.headers()?)
            .json(&body)
            .timeout(self.timeout)
            .send()
            .await
            .context("Failed to send Codex OAuth Responses API request")?;

        if !response.status().is_success() {
            let status = response.status();
            let _ = response.text().await;
            if status == reqwest::StatusCode::UNAUTHORIZED {
                anyhow::bail!(
                    "Codex OAuth session was rejected with HTTP 401. Run `codex login` to refresh ~/.codex/auth.json, then retry."
                );
            }
            anyhow::bail!("Codex OAuth Responses API HTTP error {}", status);
        }

        let sse = response
            .text()
            .await
            .context("Failed to read Codex OAuth Responses API stream")?;

        extract_responses_sse_text(&sse)
            .ok_or_else(|| anyhow::anyhow!("Invalid Codex OAuth Responses API stream format"))
    }
}

impl CodexOAuthAuth {
    fn from_codex_home() -> Result<Self> {
        let auth_path = codex_auth_path();
        validate_auth_file(&auth_path)?;
        let content = std::fs::read_to_string(&auth_path)
            .with_context(|| format!("Failed to read Codex OAuth auth file at {:?}. Run `codex login` first or set CODEX_HOME.", auth_path))?;
        Self::from_auth_json(&content)
    }

    fn from_auth_json(content: &str) -> Result<Self> {
        let auth: AuthDotJson =
            serde_json::from_str(content).context("Failed to parse Codex auth.json")?;
        let tokens = auth.tokens.context(
            "Codex auth.json does not contain OAuth tokens. Run `codex login` with ChatGPT OAuth.",
        )?;
        let claims = parse_id_token_claims(&tokens.id_token).unwrap_or(None);
        let account_id = tokens
            .account_id
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                claims
                    .as_ref()
                    .and_then(|claims| claims.chatgpt_account_id.clone())
            });
        let is_fedramp = claims
            .as_ref()
            .map(|claims| claims.chatgpt_account_is_fedramp)
            .unwrap_or(false);

        Ok(Self {
            access_token: tokens.access_token,
            account_id,
            is_fedramp,
        })
    }

    fn headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.access_token))?,
        );
        headers.insert(
            "OpenAI-Beta",
            HeaderValue::from_static("responses=2025-06-21"),
        );
        headers.insert("x-codex-client", HeaderValue::from_static("deepwiki-rs"));

        if let Some(account_id) = &self.account_id {
            headers.insert("ChatGPT-Account-ID", HeaderValue::from_str(account_id)?);
        }
        if self.is_fedramp {
            headers.insert("X-OpenAI-Fedramp", HeaderValue::from_static("true"));
        }

        Ok(headers)
    }
}

pub struct CodexOAuthExtractor<T>
where
    T: JsonSchema + Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    client: CodexOAuthClient,
    model: String,
    instructions: String,
    max_retries: u32,
    _phantom: PhantomData<T>,
}

impl<T> CodexOAuthExtractor<T>
where
    T: JsonSchema + Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    pub async fn extract(&self, prompt: &str) -> Result<T> {
        let mut last_error = None;
        for attempt in 1..=self.max_retries.max(1) {
            let enhanced_prompt = build_extraction_prompt(prompt, last_error.as_deref());
            match self
                .client
                .responses_text(&self.model, &self.instructions, &enhanced_prompt, None)
                .await
                .and_then(|response| parse_and_validate::<T>(&response, attempt))
            {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(format!("{:#}", error));
                    if attempt < self.max_retries.max(1) {
                        tokio::time::sleep(tokio::time::Duration::from_millis(3000)).await;
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "Failed after {} attempts. Last error: {}",
            self.max_retries.max(1),
            last_error.unwrap_or_else(|| "Unknown error".to_string())
        ))
    }
}

fn normalize_codex_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    match trimmed {
        "https://chatgpt.com" | "https://chat.openai.com" => {
            "https://chatgpt.com/backend-api/codex".to_string()
        }
        _ => trimmed.to_string(),
    }
}

fn codex_auth_path() -> PathBuf {
    std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".codex")
        })
        .join("auth.json")
}

fn validate_auth_file(path: &PathBuf) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("Failed to inspect Codex OAuth auth file at {:?}", path))?;

    if metadata.file_type().is_symlink() {
        anyhow::bail!(
            "Refusing to read Codex OAuth auth file symlink at {:?}",
            path
        );
    }
    if !metadata.file_type().is_file() {
        anyhow::bail!("Codex OAuth auth path is not a regular file: {:?}", path);
    }

    #[cfg(unix)]
    {
        let mode = metadata.permissions().mode();
        if mode & 0o077 != 0 {
            anyhow::bail!(
                "Refusing to read Codex OAuth auth file with group/world permissions at {:?}; run `chmod 600 {:?}`",
                path,
                path
            );
        }
    }

    Ok(())
}

fn parse_id_token_claims(id_token: &str) -> Result<Option<AuthClaims>> {
    let mut parts = id_token.split('.');
    let (_header, payload, _signature) = match (parts.next(), parts.next(), parts.next()) {
        (Some(header), Some(payload), Some(signature))
            if !header.is_empty() && !payload.is_empty() && !signature.is_empty() =>
        {
            (header, payload, signature)
        }
        _ => return Ok(None),
    };

    let payload = URL_SAFE_NO_PAD.decode(payload)?;
    let claims: IdClaims = serde_json::from_slice(&payload)?;
    Ok(claims.auth)
}

fn extract_responses_text(json: &Value) -> Option<String> {
    if let Some(text) = json.get("output_text").and_then(Value::as_str) {
        return Some(text.to_string());
    }

    let output = json.get("output")?.as_array()?;
    let mut chunks = Vec::new();
    for item in output {
        if let Some(content) = item.get("content").and_then(Value::as_array) {
            for part in content {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    chunks.push(text.to_string());
                }
            }
        }
    }

    (!chunks.is_empty()).then(|| chunks.join(""))
}

fn extract_responses_sse_text(sse: &str) -> Option<String> {
    let mut chunks = Vec::new();
    let mut completed_output: Option<String> = None;

    for block in sse.split("\n\n") {
        let mut data_lines = Vec::new();
        for line in block.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                data_lines.push(data);
            }
        }
        if data_lines.is_empty() {
            continue;
        }

        let data = data_lines.join("\n");
        let Ok(json) = serde_json::from_str::<Value>(&data) else {
            continue;
        };

        match json.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                if let Some(delta) = json.get("delta").and_then(Value::as_str) {
                    chunks.push(delta.to_string());
                }
            }
            Some("response.completed") => {
                if let Some(text) = json.get("response").and_then(extract_responses_text) {
                    completed_output = Some(text);
                }
            }
            _ => {}
        }
    }

    if !chunks.is_empty() {
        Some(chunks.join(""))
    } else {
        completed_output
    }
}

fn build_extraction_prompt(base_prompt: &str, previous_error: Option<&str>) -> String {
    let mut prompt = format!(
        "{}\n\nReturn only a valid JSON object matching the requested schema. Do not add markdown or commentary.\n",
        base_prompt
    );

    if let Some(error) = previous_error {
        prompt.push_str(&format!(
            "\nPrevious attempt failed with error: {}\nPlease fix these issues and regenerate.\n",
            error
        ));
    }

    prompt
}

fn parse_and_validate<T>(response: &str, attempt: u32) -> Result<T>
where
    T: JsonSchema + Serialize + for<'de> Deserialize<'de>,
{
    let parsed = parse_json_response(response).context("Failed to parse JSON from response")?;
    let result: T = serde_json::from_value(parsed.clone()).with_context(|| {
        let json_str =
            serde_json::to_string_pretty(&parsed).unwrap_or_else(|_| "invalid".to_string());
        format!(
            "Failed to deserialize JSON to target type on attempt {}. JSON structure: {}",
            attempt, json_str
        )
    })?;
    Ok(result)
}

fn parse_json_response(response: &str) -> Result<Value> {
    if let Ok(json) = serde_json::from_str::<Value>(response) {
        return Ok(json);
    }

    if let Some(json_str) = JSON_CODE_BLOCK_REGEX
        .captures(response)
        .and_then(|captures| captures.get(1))
        .map(|matched| matched.as_str().to_string())
    {
        if let Ok(parsed) = serde_json::from_str::<Value>(&json_str) {
            return Ok(parsed);
        }
    }

    if let Some(json_str) = extract_first_json_object(response) {
        if let Ok(parsed) = serde_json::from_str::<Value>(&json_str) {
            return Ok(parsed);
        }
    }

    serde_json::from_str::<Value>(response.trim()).with_context(|| {
        format!(
            "Response is not valid JSON: {}",
            response.chars().take(200).collect::<String>()
        )
    })
}

fn extract_first_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let mut depth = 0;
    let mut in_string = false;
    let mut escape_next = false;

    for (offset, ch) in text[start..].char_indices() {
        if escape_next {
            escape_next = false;
            continue;
        }

        match ch {
            '\\' if in_string => escape_next = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=start + offset].to_string());
                }
            }
            _ => {}
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_responses_output_text() {
        let payload = serde_json::json!({"output_text": "hello"});
        assert_eq!(extract_responses_text(&payload).unwrap(), "hello");
    }

    #[test]
    fn extracts_nested_responses_text() {
        let payload = serde_json::json!({
            "output": [{
                "content": [
                    {"type": "output_text", "text": "hel"},
                    {"type": "output_text", "text": "lo"}
                ]
            }]
        });
        assert_eq!(extract_responses_text(&payload).unwrap(), "hello");
    }

    #[test]
    fn extracts_streaming_responses_text_deltas() {
        let sse = r#"event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"hel"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"lo"}

event: response.completed
data: {"type":"response.completed","response":{"output":[{"content":[{"type":"output_text","text":"ignored fallback"}]}]}}
"#;

        assert_eq!(extract_responses_sse_text(sse).unwrap(), "hello");
    }

    #[test]
    fn parses_auth_json_without_exposing_tokens() {
        let content = r#"{
            "tokens": {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "account_id": "account-1",
                "id_token": "not.a.jwt"
            }
        }"#;
        let auth = CodexOAuthAuth::from_auth_json(content).unwrap();
        assert_eq!(auth.access_token, "access-token");
        assert_eq!(auth.account_id.as_deref(), Some("account-1"));
        assert!(!auth.is_fedramp);
    }
}
