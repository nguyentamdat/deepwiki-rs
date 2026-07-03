use crate::config::{Config, LLMProvider};
use crate::i18n::TargetLanguage;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// DeepWiki-RS - Project knowledge base generation engine powered by Rust and AI
#[derive(Parser, Debug)]
#[command(name = "Litho (deepwiki-rs)")]
#[command(
    about = "AI-based high-performance generation engine for documentation, It can intelligently analyze project structures, identify core modules, and generate professional architecture documentation."
)]
#[command(author = "Sopaco")]
#[command(version)]
pub struct Args {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Project path
    #[arg(short, long, default_value = ".")]
    pub project_path: PathBuf,

    /// Output path
    #[arg(short, long, default_value = "./litho.docs")]
    pub output_path: PathBuf,

    /// Configuration file path
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Project name
    #[arg(short, long)]
    pub name: Option<String>,

    /// Skip project preprocessing
    #[arg(long)]
    pub skip_preprocessing: bool,

    /// Skip research document generation
    #[arg(long)]
    pub skip_research: bool,

    /// Skip final document generation
    #[arg(long)]
    pub skip_documentation: bool,

    /// Enable verbose logging
    #[arg(short, long)]
    pub verbose: bool,

    /// High-efficiency model, prioritized for Litho engine's regular inference tasks
    #[arg(long)]
    pub model_efficient: Option<String>,

    /// High-quality model, prioritized for Litho engine's complex inference tasks, and as fallback when efficient fails
    #[arg(long)]
    pub model_powerful: Option<String>,

    /// LLM API base URL
    #[arg(long)]
    pub llm_api_base_url: Option<String>,

    /// LLM API KEY
    #[arg(long)]
    pub llm_api_key: Option<String>,

    /// Maximum number of tokens
    #[arg(long)]
    pub max_tokens: Option<u32>,

    /// Temperature parameter
    #[arg(long)]
    pub temperature: Option<f64>,

    /// Max parallelism parameter
    #[arg(long)]
    pub max_parallels: Option<usize>,

    /// Tool concurrency parameter
    #[arg(long)]
    pub tool_concurrency: Option<usize>,

    /// LLM Provider (openai, openai-codex, mistral, openrouter, anthropic, deepseek, gemini, ollama)
    #[arg(long)]
    pub llm_provider: Option<String>,

    /// Target language (zh, en, ja, ko, de, fr, ru, vi)
    #[arg(long)]
    pub target_language: Option<String>,

    /// Auto use report assistant to view report after generation
    #[arg(long, default_value = "false", action = clap::ArgAction::SetTrue)]
    pub disable_preset_tools: bool,

    /// Disable cache
    #[arg(long)]
    pub no_cache: bool,

    /// Force regeneration (clear cache)
    #[arg(long)]
    pub force_regenerate: bool,

    /// Code insights limit for boundary analysis
    #[arg(long)]
    pub boundary_code_limit: Option<usize>,

    /// Include source code in boundary analysis (default: true)
    #[arg(long)]
    pub boundary_include_source: Option<bool>,

    /// Show only directories when files exceed this count
    #[arg(long)]
    pub boundary_only_directories_when_files_more_than: Option<usize>,
}

/// CLI subcommands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Sync external knowledge sources (local docs, etc.)
    SyncKnowledge {
        /// Configuration file path
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Force sync even if cache is fresh
        #[arg(long)]
        force: bool,
    },
}

impl Args {
    /// Convert CLI arguments to configuration
    pub fn to_config(self) -> Config {
        // Determine target language early for proper message localization
        let target_lang = if let Some(ref lang_str) = self.target_language {
            lang_str.parse::<TargetLanguage>().unwrap_or_default()
        } else {
            TargetLanguage::default()
        };

        let mut config = if let Some(config_path) = &self.config {
            // If config file path is explicitly specified, load from that path
            let msg = target_lang
                .msg_config_read_error()
                .replace("{:?}", &format!("{:?}", config_path));
            Config::from_file(config_path).expect(&msg)
        } else {
            // If no config file is explicitly specified, try loading from default location
            let default_config_path = std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join("litho.toml");

            if default_config_path.exists() {
                let msg = target_lang
                    .msg_config_read_error()
                    .replace("{:?}", &format!("{:?}", default_config_path));
                Config::from_file(&default_config_path).expect(&msg)
            } else {
                // Default config file doesn't exist, use default values
                Config::default()
            }
        };

        let default_llm_config = crate::config::LLMConfig::default();
        let explicit_api_base_url = self.llm_api_base_url.is_some()
            || config.llm.api_base_url != default_llm_config.api_base_url;
        let explicit_models = config.llm.model_efficient != default_llm_config.model_efficient
            || config.llm.model_powerful != default_llm_config.model_powerful;
        let model_efficient_overridden = self.model_efficient.is_some();

        // Override settings from config file
        config.project_path = self.project_path.clone();
        config.output_path = self.output_path;
        config.internal_path = self.project_path.join(".litho");

        // Project name handling: CLI argument has highest priority, if CLI doesn't specify and config file doesn't have it, get_project_name() will auto-infer
        if let Some(name) = self.name {
            config.project_name = Some(name);
        }

        // Override LLM configuration
        if let Some(provider_str) = self.llm_provider {
            if let Ok(provider) = provider_str.parse::<LLMProvider>() {
                config.llm.provider = provider;
            } else {
                let msg = target_lang
                    .msg_unknown_provider()
                    .replace("{}", &provider_str);
                eprintln!("{}", msg);
            }
        }
        if let Some(llm_api_base_url) = self.llm_api_base_url {
            config.llm.api_base_url = llm_api_base_url;
        } else {
            config
                .llm
                .apply_provider_defaults(explicit_api_base_url, explicit_models);
        }
        if let Some(llm_api_key) = self.llm_api_key {
            config.llm.api_key = llm_api_key;
        }
        if let Some(model_efficient) = self.model_efficient {
            config.llm.model_efficient = model_efficient;
        }
        if let Some(model_powerful) = self.model_powerful {
            config.llm.model_powerful = model_powerful;
        } else if model_efficient_overridden && config.llm.provider.default_models().is_none() {
            config.llm.model_powerful = config.llm.model_efficient.to_string();
        }
        if let Some(max_tokens) = self.max_tokens {
            config.llm.max_tokens = max_tokens;
        }
        if let Some(temperature) = self.temperature {
            config.llm.temperature = Some(temperature);
        }
        if let Some(max_parallels) = self.max_parallels {
            config.llm.max_parallels = max_parallels;
        }
        if let Some(tool_concurrency) = self.tool_concurrency {
            config.llm.tool_concurrency = tool_concurrency;
        }
        config.llm.disable_preset_tools = self.disable_preset_tools;

        // Target language configuration
        if let Some(target_language_str) = self.target_language {
            if let Ok(target_language) = target_language_str.parse::<TargetLanguage>() {
                config.target_language = target_language;
            } else {
                let msg = target_lang
                    .msg_unknown_language()
                    .replace("{}", &target_language_str);
                eprintln!("{}", msg);
            }
        }

        // Cache configuration
        if self.no_cache {
            config.cache.enabled = false;
        }

        // Boundary analysis configuration overrides
        if let Some(code_limit) = self.boundary_code_limit {
            config.boundary_analysis.code_insights_limit = code_limit;
        }
        if let Some(include_source) = self.boundary_include_source {
            config.boundary_analysis.include_source_code = include_source;
        }
        if let Some(only_dirs_threshold) = self.boundary_only_directories_when_files_more_than {
            config
                .boundary_analysis
                .only_directories_when_files_more_than = Some(only_dirs_threshold);
        }

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempConfigFile {
        path: PathBuf,
    }

    impl TempConfigFile {
        fn empty() -> Self {
            let unique_suffix = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "deepwiki-rs-openai-codex-tests-{}-{}.toml",
                std::process::id(),
                unique_suffix
            ));

            fs::write(&path, "").unwrap();

            Self { path }
        }
    }

    impl Drop for TempConfigFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    #[test]
    fn test_to_config_applies_openai_codex_defaults_without_explicit_overrides() {
        let config_file = TempConfigFile::empty();

        let config = Args::parse_from([
            "litho",
            "--config",
            config_file.path.to_str().unwrap(),
            "--llm-provider",
            "openai-codex",
        ])
        .to_config();

        assert_eq!(config.llm.provider, LLMProvider::OpenAICodex);
        assert_eq!(config.llm.api_base_url, "https://api.openai.com/v1");
        assert_eq!(config.llm.model_efficient, "gpt-5-codex-mini");
        assert_eq!(config.llm.model_powerful, "gpt-5-codex");
    }

    #[test]
    fn test_to_config_preserves_explicit_openai_codex_overrides() {
        let config_file = TempConfigFile::empty();

        let config = Args::parse_from([
            "litho",
            "--config",
            config_file.path.to_str().unwrap(),
            "--llm-provider",
            "openai-codex",
            "--llm-api-base-url",
            "https://custom.example/v1",
            "--model-efficient",
            "custom-efficient",
            "--model-powerful",
            "custom-powerful",
        ])
        .to_config();

        assert_eq!(config.llm.provider, LLMProvider::OpenAICodex);
        assert_eq!(config.llm.api_base_url, "https://custom.example/v1");
        assert_eq!(config.llm.model_efficient, "custom-efficient");
        assert_eq!(config.llm.model_powerful, "custom-powerful");
    }

    #[test]
    fn test_to_config_keeps_codex_powerful_default_when_only_efficient_model_is_overridden() {
        let config_file = TempConfigFile::empty();

        let config = Args::parse_from([
            "litho",
            "--config",
            config_file.path.to_str().unwrap(),
            "--llm-provider",
            "openai-codex",
            "--model-efficient",
            "custom-efficient",
        ])
        .to_config();

        assert_eq!(config.llm.provider, LLMProvider::OpenAICodex);
        assert_eq!(config.llm.model_efficient, "custom-efficient");
        assert_eq!(config.llm.model_powerful, "gpt-5-codex");
    }
}
