//! Language Server Protocol implementation for BHC.
//!
//! This crate implements the [Language Server Protocol](https://microsoft.github.io/language-server-protocol/)
//! for BHC, enabling IDE features like:
//!
//! - Diagnostics (errors, warnings)
//! - Go to definition
//! - Find references
//! - Hover information
//! - Code completion
//! - Code actions
//! - Document symbols
//! - Workspace symbols
//! - Formatting
//!
//! # Architecture
//!
//! The LSP server uses a message-passing architecture:
//!
//! ```text
//!                    ┌──────────────┐
//!                    │   Editor     │
//!                    │ (VS Code,    │
//!                    │  Neovim,     │
//!                    │  etc.)       │
//!                    └──────┬───────┘
//!                           │ JSON-RPC
//!                    ┌──────▼───────┐
//!                    │  Transport   │
//!                    │  (stdio)     │
//!                    └──────┬───────┘
//!                           │
//!                    ┌──────▼───────┐
//!                    │   Server     │
//!                    │   Main Loop  │
//!                    └──────┬───────┘
//!                           │
//!         ┌─────────────────┼─────────────────┐
//!         │                 │                 │
//!  ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
//!  │  Analysis   │   │  Document   │   │   Config    │
//!  │   Engine    │   │   Manager   │   │   Manager   │
//!  └─────────────┘   └─────────────┘   └─────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use bhc_lsp::Server;
//!
//! fn main() -> anyhow::Result<()> {
//!     bhc_lsp::run()
//! }
//! ```

#![warn(missing_docs)]

pub mod analysis;
pub mod capabilities;
pub mod completion;
pub mod config;
pub mod diagnostics;
pub mod document;
pub mod handlers;
pub mod hover;
pub mod navigation;
pub mod server;
pub mod symbols;

pub use server::{run, Server};

use thiserror::Error;

/// LSP server errors.
#[derive(Debug, Error)]
pub enum LspError {
    /// Protocol error.
    #[error("protocol error: {0}")]
    Protocol(String),

    /// Analysis error.
    #[error("analysis error: {0}")]
    Analysis(String),

    /// Document not found.
    #[error("document not found: {0}")]
    DocumentNotFound(String),

    /// Invalid request.
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// LSP result type.
pub type LspResult<T> = Result<T, LspError>;
