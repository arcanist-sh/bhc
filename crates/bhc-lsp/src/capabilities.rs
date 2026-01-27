//! Server capabilities.
//!
//! Defines what features the LSP server supports.

use lsp_types::{
    CompletionOptions, HoverProviderCapability, OneOf, SaveOptions, ServerCapabilities,
    TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions,
    TextDocumentSyncSaveOptions, WorkDoneProgressOptions,
};

/// Get the server capabilities.
pub fn server_capabilities() -> ServerCapabilities {
    ServerCapabilities {
        // Text document sync
        text_document_sync: Some(TextDocumentSyncCapability::Options(
            TextDocumentSyncOptions {
                open_close: Some(true),
                change: Some(TextDocumentSyncKind::INCREMENTAL),
                will_save: Some(false),
                will_save_wait_until: Some(false),
                save: Some(TextDocumentSyncSaveOptions::SaveOptions(SaveOptions {
                    include_text: Some(false),
                })),
            },
        )),

        // Hover support
        hover_provider: Some(HoverProviderCapability::Simple(true)),

        // Completion support
        completion_provider: Some(CompletionOptions {
            resolve_provider: Some(true),
            trigger_characters: Some(vec![".".to_string(), "::".to_string()]),
            all_commit_characters: None,
            work_done_progress_options: WorkDoneProgressOptions::default(),
            completion_item: None,
        }),

        // Go to definition
        definition_provider: Some(OneOf::Left(true)),

        // Find references
        references_provider: Some(OneOf::Left(true)),

        // Document symbols
        document_symbol_provider: Some(OneOf::Left(true)),

        // Workspace symbols
        workspace_symbol_provider: Some(OneOf::Left(true)),

        // Formatting
        document_formatting_provider: Some(OneOf::Left(true)),

        // Code actions (quick fixes, refactorings)
        // TODO: Enable when implemented
        code_action_provider: None,

        // Rename support
        // TODO: Enable when implemented
        rename_provider: None,

        // Signature help
        // TODO: Enable when implemented
        signature_help_provider: None,

        // Document highlight
        // TODO: Enable when implemented
        document_highlight_provider: None,

        // Folding ranges
        // TODO: Enable when implemented
        folding_range_provider: None,

        // Semantic tokens
        // TODO: Enable when implemented
        semantic_tokens_provider: None,

        // Inlay hints
        // TODO: Enable when implemented
        inlay_hint_provider: None,

        ..Default::default()
    }
}
