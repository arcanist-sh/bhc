//! Deeply nested input must produce a diagnostic, not a stack overflow.

use bhc_parser::{parse_expr, parse_module};
use bhc_span::FileId;

const DEPTH: usize = 20_000;

#[test]
fn deeply_nested_parens_error_gracefully() {
    let src = format!("{}1{}", "(".repeat(DEPTH), ")".repeat(DEPTH));
    let (_expr, diagnostics) = parse_expr(&src, FileId::new(0));
    assert!(
        diagnostics.iter().any(|d| d.message.contains("nesting")),
        "expected a recursion-limit diagnostic, got: {diagnostics:?}"
    );
}

#[test]
fn deeply_nested_module_expr_errors_gracefully() {
    let src = format!("x = {}1{}\n", "(".repeat(DEPTH), ")".repeat(DEPTH));
    let (_module, diagnostics) = parse_module(&src, FileId::new(0));
    assert!(
        !diagnostics.is_empty(),
        "expected diagnostics for deeply nested input"
    );
}
