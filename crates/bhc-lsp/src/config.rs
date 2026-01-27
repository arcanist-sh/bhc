//! Server configuration.
//!
//! Handles server settings from the client.

use serde::{Deserialize, Serialize};

/// Server configuration.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Config {
    /// Diagnostics settings.
    #[serde(default)]
    pub diagnostics: DiagnosticsConfig,

    /// Completion settings.
    #[serde(default)]
    pub completion: CompletionConfig,

    /// Formatting settings.
    #[serde(default)]
    pub formatting: FormattingConfig,

    /// Inlay hints settings.
    #[serde(default)]
    pub inlay_hints: InlayHintsConfig,
}

impl Config {
    /// Update config from a JSON value.
    pub fn update_from_value(&self, _value: &serde_json::Value) -> Result<(), serde_json::Error> {
        // Parse and merge settings
        // In a real implementation, we'd have mutable fields wrapped in RwLock
        Ok(())
    }
}

/// Diagnostics configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiagnosticsConfig {
    /// Enable diagnostics.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Disable specific diagnostics by code.
    #[serde(default)]
    pub disabled: Vec<String>,

    /// Maximum number of diagnostics to report per file.
    #[serde(default = "default_max_diagnostics")]
    pub max_per_file: usize,

    /// Enable warnings.
    #[serde(default = "default_true")]
    pub warnings: bool,
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            disabled: Vec::new(),
            max_per_file: 100,
            warnings: true,
        }
    }
}

/// Completion configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionConfig {
    /// Enable completions.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Auto-import on completion.
    #[serde(default = "default_true")]
    pub auto_import: bool,

    /// Include snippets in completions.
    #[serde(default = "default_true")]
    pub snippets: bool,

    /// Maximum number of completions.
    #[serde(default = "default_max_completions")]
    pub max_results: usize,
}

impl Default for CompletionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_import: true,
            snippets: true,
            max_results: 100,
        }
    }
}

/// Formatting configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FormattingConfig {
    /// Enable formatting.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Formatter to use.
    #[serde(default = "default_formatter")]
    pub formatter: String,

    /// Line width.
    #[serde(default = "default_line_width")]
    pub line_width: usize,

    /// Indent width.
    #[serde(default = "default_indent")]
    pub indent_width: usize,

    /// Use tabs instead of spaces.
    #[serde(default)]
    pub use_tabs: bool,
}

impl Default for FormattingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            formatter: "bhc".to_string(),
            line_width: 80,
            indent_width: 2,
            use_tabs: false,
        }
    }
}

/// Inlay hints configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InlayHintsConfig {
    /// Enable inlay hints.
    #[serde(default)]
    pub enabled: bool,

    /// Show type hints.
    #[serde(default = "default_true")]
    pub type_hints: bool,

    /// Show parameter hints.
    #[serde(default = "default_true")]
    pub parameter_hints: bool,

    /// Maximum hint length.
    #[serde(default = "default_max_hint_length")]
    pub max_length: usize,
}

impl Default for InlayHintsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            type_hints: true,
            parameter_hints: true,
            max_length: 25,
        }
    }
}

// Default value functions
fn default_true() -> bool {
    true
}

fn default_max_diagnostics() -> usize {
    100
}

fn default_max_completions() -> usize {
    100
}

fn default_formatter() -> String {
    "bhc".to_string()
}

fn default_line_width() -> usize {
    80
}

fn default_indent() -> usize {
    2
}

fn default_max_hint_length() -> usize {
    25
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.diagnostics.enabled);
        assert!(config.completion.enabled);
        assert!(config.formatting.enabled);
    }

    #[test]
    fn test_deserialize_config() {
        let json = r#"
        {
            "diagnostics": {
                "enabled": true,
                "warnings": false
            },
            "completion": {
                "auto_import": false
            }
        }
        "#;

        let config: Config = serde_json::from_str(json).unwrap();
        assert!(!config.diagnostics.warnings);
        assert!(!config.completion.auto_import);
    }
}
