//! Regression tests for view patterns appearing as tuple / list elements.
//!
//! A view pattern (`expr -> pat`) used as an element of a tuple or list
//! pattern — e.g. `(l, T.uncons -> Just ('>', r))` — was only recognized when
//! the view pattern was the *sole* content of the parentheses. As a tuple
//! element it was parsed by the general pattern parser, which stops at `->`,
//! so the closing delimiter check failed and error recovery silently dropped
//! the *entire enclosing declaration*. This surfaced downstream as a spurious
//! `unbound variable` for the dropped function (e.g. Pandoc's
//! `Text.Pandoc.Readers.DokuWiki:splitInterwiki`).
//!
//! Each test asserts that both the function containing the view pattern and a
//! following function are present in the parsed declarations (i.e. nothing was
//! dropped) and that parsing produced no errors.

use bhc_parser::parse_module;
use bhc_span::FileId;

fn parse(src: &str) -> (usize, usize) {
    let (module, diagnostics) = parse_module(src, FileId::new(0));
    let errors = diagnostics
        .iter()
        .filter(|d| d.severity == bhc_diagnostics::Severity::Error)
        .count();
    // Count value bindings (FunBind) among the top-level declarations.
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
fn view_pattern_as_second_tuple_element() {
    // `zog`'s body has a view pattern nested in a tuple case-alt pattern.
    // Before the fix, `zog`'s FunBind was dropped and `zog` reported unbound.
    let src = r#"
{-# LANGUAGE ViewPatterns #-}
module M where
zog :: (Int, Maybe Int) -> Int
zog x = case x of
    (l, id -> Just r) -> l + r
    _ -> 0
caller :: (Int, Maybe Int) -> Int
caller q = zog q
"#;
    let (fun_binds, errors) = parse(src);
    assert_eq!(errors, 0, "expected no parse errors");
    assert_eq!(fun_binds, 2, "both zog and caller must survive parsing");
}

#[test]
fn view_pattern_as_first_tuple_element() {
    let src = r#"
{-# LANGUAGE ViewPatterns #-}
module M where
zog :: (Maybe Int, Int) -> Int
zog x = case x of
    (id -> Just r, l) -> l + r
    _ -> 0
caller :: (Maybe Int, Int) -> Int
caller q = zog q
"#;
    let (fun_binds, errors) = parse(src);
    assert_eq!(errors, 0, "expected no parse errors");
    assert_eq!(fun_binds, 2, "both zog and caller must survive parsing");
}

#[test]
fn view_pattern_as_list_element() {
    let src = r#"
{-# LANGUAGE ViewPatterns #-}
module M where
zog :: [Maybe Int] -> Int
zog x = case x of
    [l, id -> Just r] -> r
    _ -> 0
caller :: [Maybe Int] -> Int
caller q = zog q
"#;
    let (fun_binds, errors) = parse(src);
    assert_eq!(errors, 0, "expected no parse errors");
    assert_eq!(fun_binds, 2, "both zog and caller must survive parsing");
}

#[test]
fn applied_view_pattern_as_tuple_element() {
    // View function applied to an argument, as a tuple element: `(l, f k -> p)`.
    let src = r#"
{-# LANGUAGE ViewPatterns #-}
module M where
lu :: Int -> Maybe Int -> Maybe Int
lu _ m = m
zog :: (Int, Maybe Int) -> Int
zog x = case x of
    (l, lu 0 -> Just r) -> l + r
    _ -> 0
caller :: (Int, Maybe Int) -> Int
caller q = zog q
"#;
    let (fun_binds, errors) = parse(src);
    assert_eq!(errors, 0, "expected no parse errors");
    assert_eq!(fun_binds, 3, "lu, zog and caller must survive parsing");
}

#[test]
fn standalone_view_pattern_still_parses() {
    // Ensure the refactor didn't break the standalone `(e -> p)` form.
    let src = r#"
{-# LANGUAGE ViewPatterns #-}
module M where
zog :: Maybe Int -> Int
zog (id -> Just r) = r
zog _ = 0
"#;
    let (fun_binds, errors) = parse(src);
    assert_eq!(errors, 0, "expected no parse errors");
    assert_eq!(fun_binds, 1, "zog must parse");
}
