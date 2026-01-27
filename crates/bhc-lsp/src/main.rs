//! BHC Language Server
//!
//! This is the main entry point for the BHC Language Server Protocol implementation.
//!
//! # Usage
//!
//! ```bash
//! bhc-lsp
//! ```
//!
//! The server communicates over stdio using JSON-RPC.

use anyhow::Result;

fn main() -> Result<()> {
    bhc_lsp::run()
}
