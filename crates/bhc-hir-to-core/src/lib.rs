//! # HIR to Core Lowering
//!
//! This crate transforms typed HIR (High-Level IR) into Core IR, the main
//! intermediate representation used for optimization.
//!
//! ## Key Transformations
//!
//! - **Pattern compilation**: Multi-argument lambdas and pattern matching
//!   are compiled into explicit case expressions
//! - **Binding analysis**: Let bindings are analyzed for mutual recursion
//! - **Guard expansion**: Pattern guards become nested conditionals
//! - **Type erasure**: Type annotations are preserved but simplified

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

mod binding;
mod context;
mod expr;
mod pattern;

use bhc_core::CoreModule;
use bhc_hir::Module as HirModule;
use bhc_span::Span;
use thiserror::Error;

pub use context::LowerContext;

/// Errors that can occur during HIR to Core lowering.
#[derive(Debug, Error)]
pub enum LowerError {
    /// An internal invariant was violated.
    #[error("internal error: {0}")]
    Internal(String),

    /// Pattern compilation failed.
    #[error("pattern compilation failed at {span:?}: {message}")]
    PatternError {
        /// Error message.
        message: String,
        /// Source location.
        span: Span,
    },

    /// Multiple errors occurred.
    #[error("multiple errors")]
    Multiple(Vec<LowerError>),
}

/// Result type for lowering operations.
pub type LowerResult<T> = Result<T, LowerError>;

/// Lower a HIR module to Core IR.
///
/// This is the main entry point for the HIR to Core transformation.
///
/// # Arguments
///
/// * `module` - The typed HIR module to lower
///
/// # Returns
///
/// A `CoreModule` containing the lowered bindings.
///
/// # Errors
///
/// Returns `LowerError` if lowering fails due to internal errors or
/// unsupported constructs.
pub fn lower_module(module: &HirModule) -> LowerResult<CoreModule> {
    let mut ctx = LowerContext::new();
    ctx.lower_module(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_intern::Symbol;

    #[test]
    fn test_lower_empty_module() {
        let module = HirModule {
            name: Symbol::intern("Test"),
            exports: None,
            imports: vec![],
            items: vec![],
            span: Span::default(),
        };

        let result = lower_module(&module);
        assert!(result.is_ok());
        let core_module = result.unwrap();
        assert!(core_module.bindings.is_empty());
    }
}
