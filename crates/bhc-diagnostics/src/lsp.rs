//! Language Server Protocol (LSP) integration for BHC diagnostics.
//!
//! This module provides conversion from BHC's internal diagnostic format
//! to the LSP diagnostic format, enabling rich IDE integration.
//!
//! ## Features
//!
//! - Convert diagnostics to LSP format
//! - Generate code actions from suggestions
//! - Provide hover information for error spans
//!
//! ## Example
//!
//! ```ignore
//! use bhc_diagnostics::{Diagnostic, lsp};
//!
//! let diagnostic = Diagnostic::error("type mismatch")
//!     .with_code("E0001");
//!
//! let lsp_diag = lsp::to_lsp_diagnostic(&diagnostic, &source_map);
//! let code_actions = lsp::to_code_actions(&diagnostic, &source_map, &uri);
//! ```

use crate::{Applicability, Diagnostic, Severity, SourceMap, Suggestion};
use bhc_span::SourceFile;
use serde::{Deserialize, Serialize};

/// LSP diagnostic severity levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LspSeverity {
    /// Error - prevents compilation.
    Error = 1,
    /// Warning - doesn't prevent compilation.
    Warning = 2,
    /// Information - general info.
    Information = 3,
    /// Hint - suggestion for improvement.
    Hint = 4,
}

impl From<Severity> for LspSeverity {
    fn from(severity: Severity) -> Self {
        match severity {
            Severity::Bug | Severity::Error => Self::Error,
            Severity::Warning => Self::Warning,
            Severity::Note => Self::Information,
            Severity::Help => Self::Hint,
        }
    }
}

/// LSP position (0-indexed).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspPosition {
    /// Line number (0-indexed).
    pub line: u32,
    /// Character offset (0-indexed, UTF-16 code units).
    pub character: u32,
}

impl LspPosition {
    /// Create a new position.
    #[must_use]
    pub fn new(line: u32, character: u32) -> Self {
        Self { line, character }
    }
}

/// LSP range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspRange {
    /// Start position.
    pub start: LspPosition,
    /// End position.
    pub end: LspPosition,
}

impl LspRange {
    /// Create a new range.
    #[must_use]
    pub fn new(start: LspPosition, end: LspPosition) -> Self {
        Self { start, end }
    }

    /// Create a single-point range.
    #[must_use]
    pub fn point(pos: LspPosition) -> Self {
        Self {
            start: pos,
            end: pos,
        }
    }
}

/// LSP location (URI + range).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspLocation {
    /// Document URI.
    pub uri: String,
    /// Range within the document.
    pub range: LspRange,
}

/// LSP diagnostic tag for additional categorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LspDiagnosticTag {
    /// Code is unnecessary (e.g., unused variable).
    Unnecessary = 1,
    /// Code is deprecated.
    Deprecated = 2,
}

/// Related information for an LSP diagnostic.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspRelatedInformation {
    /// Location of related information.
    pub location: LspLocation,
    /// Message describing the relationship.
    pub message: String,
}

/// LSP diagnostic code structure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LspDiagnosticCode {
    /// String code (e.g., "E0001").
    String(String),
    /// Numeric code.
    Number(i32),
}

/// An LSP diagnostic.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspDiagnostic {
    /// Range of the diagnostic.
    pub range: LspRange,
    /// Severity of the diagnostic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<LspSeverity>,
    /// Diagnostic code.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<LspDiagnosticCode>,
    /// Human-readable code description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_description: Option<LspCodeDescription>,
    /// Source of the diagnostic (e.g., "bhc").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Main diagnostic message.
    pub message: String,
    /// Diagnostic tags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<LspDiagnosticTag>>,
    /// Related information from other locations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub related_information: Option<Vec<LspRelatedInformation>>,
    /// Custom data (for code actions).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Code description with href.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspCodeDescription {
    /// URI for more information.
    pub href: String,
}

/// LSP text edit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspTextEdit {
    /// Range to replace.
    pub range: LspRange,
    /// New text to insert.
    pub new_text: String,
}

/// LSP workspace edit for a single document.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspTextDocumentEdit {
    /// Document identifier.
    pub text_document: LspVersionedTextDocumentIdentifier,
    /// Edits to apply.
    pub edits: Vec<LspTextEdit>,
}

/// Versioned text document identifier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LspVersionedTextDocumentIdentifier {
    /// Document URI.
    pub uri: String,
    /// Document version.
    pub version: Option<i32>,
}

/// LSP workspace edit.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspWorkspaceEdit {
    /// Document edits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_changes: Option<Vec<LspTextDocumentEdit>>,
}

/// Code action kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodeActionKind(pub String);

impl CodeActionKind {
    /// Quick fix action.
    pub const QUICKFIX: &'static str = "quickfix";
    /// Refactor action.
    pub const REFACTOR: &'static str = "refactor";
    /// Source action.
    pub const SOURCE: &'static str = "source";
}

/// LSP code action.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspCodeAction {
    /// Title of the action.
    pub title: String,
    /// Kind of action.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Diagnostics this action resolves.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<LspDiagnostic>>,
    /// Is this the preferred action?
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_preferred: Option<bool>,
    /// Edit to apply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edit: Option<LspWorkspaceEdit>,
}

/// Hover information content.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspHover {
    /// Hover contents.
    pub contents: LspMarkupContent,
    /// Range this hover applies to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<LspRange>,
}

/// Markup content for hover/completions.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LspMarkupContent {
    /// Kind of markup.
    pub kind: String,
    /// Markup value.
    pub value: String,
}

impl LspMarkupContent {
    /// Create plaintext content.
    #[must_use]
    pub fn plaintext(value: impl Into<String>) -> Self {
        Self {
            kind: "plaintext".to_string(),
            value: value.into(),
        }
    }

    /// Create markdown content.
    #[must_use]
    pub fn markdown(value: impl Into<String>) -> Self {
        Self {
            kind: "markdown".to_string(),
            value: value.into(),
        }
    }
}

// ============================================================
// Conversion Functions
// ============================================================

/// Convert a BHC span to an LSP range using the source file.
#[must_use]
pub fn span_to_range(file: &SourceFile, span: bhc_span::Span) -> LspRange {
    if span.is_dummy() {
        return LspRange::default();
    }

    let start_loc = file.lookup_line_col(span.lo);
    let end_loc = file.lookup_line_col(span.hi);

    LspRange {
        start: LspPosition {
            // LSP uses 0-indexed lines
            line: start_loc.line.saturating_sub(1) as u32,
            character: start_loc.col.saturating_sub(1) as u32,
        },
        end: LspPosition {
            line: end_loc.line.saturating_sub(1) as u32,
            character: end_loc.col.saturating_sub(1) as u32,
        },
    }
}

/// Convert a BHC diagnostic to an LSP diagnostic.
///
/// # Arguments
///
/// * `diagnostic` - The BHC diagnostic to convert
/// * `source_map` - The source map for location lookup
///
/// # Returns
///
/// An LSP diagnostic suitable for sending to a language client.
#[must_use]
pub fn to_lsp_diagnostic(diagnostic: &Diagnostic, source_map: &SourceMap) -> Option<LspDiagnostic> {
    // Find the primary label's range
    let primary_label = diagnostic.labels.iter().find(|l| l.primary)?;
    let file = source_map.get_file(primary_label.span.file)?;
    let range = span_to_range(file, primary_label.span.span);

    // Build the message with notes
    let mut message = diagnostic.message.clone();
    if !primary_label.message.is_empty() {
        message.push_str("\n\n");
        message.push_str(&primary_label.message);
    }
    for note in &diagnostic.notes {
        message.push_str("\n\nnote: ");
        message.push_str(note);
    }

    // Build related information from secondary labels
    let related_information: Vec<LspRelatedInformation> = diagnostic
        .labels
        .iter()
        .filter(|l| !l.primary)
        .filter_map(|label| {
            let file = source_map.get_file(label.span.file)?;
            Some(LspRelatedInformation {
                location: LspLocation {
                    uri: format!("file://{}", file.name),
                    range: span_to_range(file, label.span.span),
                },
                message: label.message.clone(),
            })
        })
        .collect();

    // Build code description for explain link
    let code_description = diagnostic.code.as_ref().map(|code| LspCodeDescription {
        href: format!("https://bhc.dev/errors/{code}"),
    });

    // Detect diagnostic tags
    let tags = detect_diagnostic_tags(diagnostic);

    Some(LspDiagnostic {
        range,
        severity: Some(diagnostic.severity.into()),
        code: diagnostic
            .code
            .as_ref()
            .map(|c| LspDiagnosticCode::String(c.clone())),
        code_description,
        source: Some("bhc".to_string()),
        message,
        tags: if tags.is_empty() { None } else { Some(tags) },
        related_information: if related_information.is_empty() {
            None
        } else {
            Some(related_information)
        },
        data: None,
    })
}

/// Detect diagnostic tags based on error code and message.
fn detect_diagnostic_tags(diagnostic: &Diagnostic) -> Vec<LspDiagnosticTag> {
    let mut tags = Vec::new();

    // Check for unused warnings
    if let Some(code) = &diagnostic.code {
        if code == "W0001" || diagnostic.message.to_lowercase().contains("unused") {
            tags.push(LspDiagnosticTag::Unnecessary);
        }
    }

    // Check for deprecated
    if diagnostic.message.to_lowercase().contains("deprecated") {
        tags.push(LspDiagnosticTag::Deprecated);
    }

    tags
}

/// Convert BHC diagnostic suggestions to LSP code actions.
///
/// # Arguments
///
/// * `diagnostic` - The BHC diagnostic with suggestions
/// * `source_map` - The source map for location lookup
/// * `uri` - The document URI
/// * `version` - Optional document version
///
/// # Returns
///
/// A list of code actions that can be applied.
#[must_use]
pub fn to_code_actions(
    diagnostic: &Diagnostic,
    source_map: &SourceMap,
    uri: &str,
    version: Option<i32>,
) -> Vec<LspCodeAction> {
    diagnostic
        .suggestions
        .iter()
        .filter_map(|suggestion| {
            suggestion_to_code_action(suggestion, diagnostic, source_map, uri, version)
        })
        .collect()
}

/// Convert a single suggestion to a code action.
fn suggestion_to_code_action(
    suggestion: &Suggestion,
    diagnostic: &Diagnostic,
    source_map: &SourceMap,
    uri: &str,
    version: Option<i32>,
) -> Option<LspCodeAction> {
    let file = source_map.get_file(suggestion.span.file)?;
    let range = span_to_range(file, suggestion.span.span);

    // Determine if this is the preferred action
    let is_preferred = matches!(suggestion.applicability, Applicability::MachineApplicable);

    // Create the text edit
    let edit = LspTextEdit {
        range,
        new_text: suggestion.replacement.clone(),
    };

    // Create workspace edit
    let workspace_edit = LspWorkspaceEdit {
        document_changes: Some(vec![LspTextDocumentEdit {
            text_document: LspVersionedTextDocumentIdentifier {
                uri: uri.to_string(),
                version,
            },
            edits: vec![edit],
        }]),
    };

    // Convert the diagnostic for association
    let lsp_diag = to_lsp_diagnostic(diagnostic, source_map);

    Some(LspCodeAction {
        title: suggestion.message.clone(),
        kind: Some(CodeActionKind::QUICKFIX.to_string()),
        diagnostics: lsp_diag.map(|d| vec![d]),
        is_preferred: Some(is_preferred),
        edit: Some(workspace_edit),
    })
}

/// Generate hover information for a diagnostic at a given position.
///
/// # Arguments
///
/// * `diagnostic` - The diagnostic to show hover for
/// * `source_map` - The source map for location lookup
///
/// # Returns
///
/// Hover information with explanation and suggestions.
#[must_use]
pub fn to_hover(diagnostic: &Diagnostic, source_map: &SourceMap) -> Option<LspHover> {
    let primary_label = diagnostic.labels.iter().find(|l| l.primary)?;
    let file = source_map.get_file(primary_label.span.file)?;
    let range = span_to_range(file, primary_label.span.span);

    // Build markdown content
    let mut content = String::new();

    // Header with severity and code
    content.push_str("**");
    content.push_str(diagnostic.severity.label());
    if let Some(code) = &diagnostic.code {
        content.push('[');
        content.push_str(code);
        content.push(']');
    }
    content.push_str("**: ");
    content.push_str(&diagnostic.message);
    content.push_str("\n\n");

    // Primary label message
    if !primary_label.message.is_empty() {
        content.push_str(&primary_label.message);
        content.push_str("\n\n");
    }

    // Notes
    for note in &diagnostic.notes {
        content.push_str("*Note*: ");
        content.push_str(note);
        content.push_str("\n\n");
    }

    // Suggestions
    if !diagnostic.suggestions.is_empty() {
        content.push_str("---\n\n");
        content.push_str("**Suggestions:**\n\n");
        for suggestion in &diagnostic.suggestions {
            content.push_str("- ");
            content.push_str(&suggestion.message);
            if !suggestion.replacement.is_empty() {
                content.push_str("\n  ```haskell\n  ");
                content.push_str(&suggestion.replacement);
                content.push_str("\n  ```");
            }
            content.push('\n');
        }
    }

    // Error explanation link
    if let Some(code) = &diagnostic.code {
        content.push_str("\n---\n\n");
        content.push_str(&format!(
            "[View explanation for {code}](https://bhc.dev/errors/{code})"
        ));
    }

    Some(LspHover {
        contents: LspMarkupContent::markdown(content),
        range: Some(range),
    })
}

/// Batch convert multiple diagnostics to LSP format.
///
/// # Arguments
///
/// * `diagnostics` - The BHC diagnostics to convert
/// * `source_map` - The source map for location lookup
///
/// # Returns
///
/// A list of LSP diagnostics.
#[must_use]
pub fn to_lsp_diagnostics(
    diagnostics: &[Diagnostic],
    source_map: &SourceMap,
) -> Vec<LspDiagnostic> {
    diagnostics
        .iter()
        .filter_map(|d| to_lsp_diagnostic(d, source_map))
        .collect()
}

/// Publish diagnostics notification for LSP.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublishDiagnosticsParams {
    /// Document URI.
    pub uri: String,
    /// Document version (if known).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<i32>,
    /// Diagnostics for this document.
    pub diagnostics: Vec<LspDiagnostic>,
}

/// Create a publish diagnostics notification.
#[must_use]
pub fn publish_diagnostics(
    uri: &str,
    diagnostics: &[Diagnostic],
    source_map: &SourceMap,
    version: Option<i32>,
) -> PublishDiagnosticsParams {
    PublishDiagnosticsParams {
        uri: uri.to_string(),
        version,
        diagnostics: to_lsp_diagnostics(diagnostics, source_map),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FullSpan;
    use bhc_span::{FileId, Span};

    fn create_test_source_map() -> SourceMap {
        let mut sm = SourceMap::new();
        sm.add_file("test.hs".into(), "foo = x + 1\nbar = y".into());
        sm
    }

    #[test]
    fn test_to_lsp_diagnostic() {
        let sm = create_test_source_map();
        let span = FullSpan::new(FileId::new(0), Span::from_raw(6, 7));

        let diag = Diagnostic::error("undefined variable `x`")
            .with_code("E0003")
            .with_label(span, "not found in this scope");

        let lsp_diag = to_lsp_diagnostic(&diag, &sm).unwrap();

        assert_eq!(lsp_diag.severity, Some(LspSeverity::Error));
        assert!(lsp_diag.message.contains("undefined variable"));
        assert_eq!(
            lsp_diag.code,
            Some(LspDiagnosticCode::String("E0003".to_string()))
        );
        assert_eq!(lsp_diag.source, Some("bhc".to_string()));
    }

    #[test]
    fn test_to_code_actions() {
        let sm = create_test_source_map();
        let span = FullSpan::new(FileId::new(0), Span::from_raw(6, 7));

        let diag = Diagnostic::error("undefined variable")
            .with_code("E0003")
            .with_label(span, "not found")
            .with_suggestion(Suggestion::new(
                "did you mean `y`?",
                span,
                "y",
                Applicability::MachineApplicable,
            ));

        let actions = to_code_actions(&diag, &sm, "file:///test.hs", Some(1));

        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].title, "did you mean `y`?");
        assert_eq!(actions[0].is_preferred, Some(true));
    }

    #[test]
    fn test_to_hover() {
        let sm = create_test_source_map();
        let span = FullSpan::new(FileId::new(0), Span::from_raw(6, 7));

        let diag = Diagnostic::error("type mismatch")
            .with_code("E0001")
            .with_label(span, "expected Int")
            .with_note("consider using `fromIntegral`");

        let hover = to_hover(&diag, &sm).unwrap();

        assert!(hover.contents.value.contains("type mismatch"));
        assert!(hover.contents.value.contains("E0001"));
        assert!(hover.contents.value.contains("fromIntegral"));
    }

    #[test]
    fn test_diagnostic_tags_unused() {
        let sm = create_test_source_map();
        let span = FullSpan::new(FileId::new(0), Span::from_raw(0, 3));

        let diag = Diagnostic::warning("unused variable `foo`")
            .with_code("W0001")
            .with_label(span, "this variable is never used");

        let lsp_diag = to_lsp_diagnostic(&diag, &sm).unwrap();

        assert!(lsp_diag.tags.is_some());
        assert!(lsp_diag
            .tags
            .unwrap()
            .contains(&LspDiagnosticTag::Unnecessary));
    }

    #[test]
    fn test_severity_conversion() {
        assert_eq!(LspSeverity::from(Severity::Error), LspSeverity::Error);
        assert_eq!(LspSeverity::from(Severity::Bug), LspSeverity::Error);
        assert_eq!(LspSeverity::from(Severity::Warning), LspSeverity::Warning);
        assert_eq!(LspSeverity::from(Severity::Note), LspSeverity::Information);
        assert_eq!(LspSeverity::from(Severity::Help), LspSeverity::Hint);
    }

    #[test]
    fn test_span_to_range() {
        let sm = create_test_source_map();
        let file = sm.get_file(FileId::new(0)).unwrap();

        // "x" at position 6-7 on first line
        let range = span_to_range(file, Span::from_raw(6, 7));

        // LSP is 0-indexed
        assert_eq!(range.start.line, 0);
        assert_eq!(range.end.line, 0);
    }

    #[test]
    fn test_publish_diagnostics() {
        let sm = create_test_source_map();
        let span = FullSpan::new(FileId::new(0), Span::from_raw(6, 7));

        let diags = vec![Diagnostic::error("test error")
            .with_code("E0001")
            .with_label(span, "here")];

        let params = publish_diagnostics("file:///test.hs", &diags, &sm, Some(1));

        assert_eq!(params.uri, "file:///test.hs");
        assert_eq!(params.version, Some(1));
        assert_eq!(params.diagnostics.len(), 1);
    }
}
