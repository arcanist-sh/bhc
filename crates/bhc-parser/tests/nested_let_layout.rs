//! Regression test for nested `let … in` layout dropping a declaration.
//!
//! `let a = let x = 1 in x in a`, written with the inner `let`'s `in` on its
//! own line, corrupted the layout: the inner `in` dedented past the inner
//! `let` (closing it via the indentation rule) and the lexer's dedicated `in`
//! handler *also* closed the outer `let` block, emitting an unbalanced extra
//! virtual `}`. The resulting parse error made error recovery silently discard
//! the whole enclosing declaration, which then surfaced downstream as a
//! spurious `unbound variable` (e.g. Pandoc's `Writers.Textile:showDim`).

use bhc_parser::parse_module;
use bhc_span::FileId;

fn fun_binds_and_errors(src: &str) -> (usize, usize) {
    let (module, diagnostics) = parse_module(src, FileId::new(0));
    let errors = diagnostics
        .iter()
        .filter(|d| d.severity == bhc_diagnostics::Severity::Error)
        .count();
    let fun_binds = module
        .as_ref()
        .map(|m| {
            m.decls
                .iter()
                .filter(|d| matches!(d, bhc_ast::Decl::FunBind(_)))
                .count()
        })
        .unwrap_or(0);
    (fun_binds, errors)
}

#[test]
fn nested_let_in_under_inner_let() {
    // `in` aligned under the inner `let` — the common idiom.
    let src = "module M where\nh :: String\nh = let a = let x = \"y\"\n            in x\n    in a\ncaller :: String\ncaller = h\n";
    let (fun_binds, errors) = fun_binds_and_errors(src);
    assert_eq!(errors, 0, "expected no parse errors");
    assert_eq!(fun_binds, 2, "both h and caller must survive parsing");
}

#[test]
fn nested_let_in_under_outer_binding() {
    // `in` aligned under the outer binding column.
    let src = "module M where\nh :: String\nh = let a = let x = \"y\"\n        in x\n    in a\ncaller :: String\ncaller = h\n";
    let (fun_binds, errors) = fun_binds_and_errors(src);
    assert_eq!(errors, 0, "expected no parse errors");
    assert_eq!(fun_binds, 2, "both h and caller must survive parsing");
}

#[test]
fn let_bound_function_with_nested_let_body() {
    // The original Pandoc shape: a let-bound function whose RHS is `let … in`,
    // referenced by a sibling binding.
    let src = "module M where\nh :: Int -> String\nh opts =\n  let showDim dir = let toCss str = Just str\n                    in case dir of\n                         1 -> toCss \"x\"\n                         _ -> Nothing\n      styles = case showDim (1 :: Int) of\n                 Just w -> w\n                 _ -> \"\"\n  in styles\ncaller :: Int -> String\ncaller q = h q\n";
    let (fun_binds, errors) = fun_binds_and_errors(src);
    assert_eq!(errors, 0, "expected no parse errors");
    assert_eq!(fun_binds, 2, "both h and caller must survive parsing");
}
