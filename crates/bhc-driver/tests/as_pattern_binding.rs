//! Regression: a `let`/`where` binding whose LHS is an as-pattern
//! (`name@subpat = expr`, e.g. `key@(Key k) = ...`) must parse as a pattern
//! binding, not a function head.
//!
//! Before the fix, the declaration parser handled infix-constructor and
//! infix-operator LHSs but not a leading `name@`, so the `@` was treated as a
//! bogus function argument, the parse failed, and the enclosing binding was
//! silently dropped — surfacing as "unbound variable" at its use sites.
//! Text.Pandoc.Readers.RST's `resolveReferences` has
//!   let key@(Key key') = toKey $ stripFirstAndLast ref
//! inside a guard's `do` block, which dropped the whole function.

use bhc_driver::Compiler;
use std::io::Write;

fn check_ok(source: &str) {
    let mut file = tempfile::Builder::new()
        .suffix(".hs")
        .tempfile()
        .expect("temp");
    file.write_all(source.as_bytes()).expect("write");
    let path = camino::Utf8Path::from_path(file.path()).expect("utf8");
    let compiler = Compiler::with_defaults().expect("compiler");
    assert!(
        compiler.check_file(path).is_ok(),
        "expected the module to check"
    );
}

#[test]
fn as_pattern_let_in_do_block() {
    check_ok(concat!(
        "module M where\n",
        "data K = K Int\n",
        "g :: Int -> IO Int\n",
        "g x = do\n",
        "  let key@(K k) = K x\n",
        "  return x\n",
        "usesG :: Int -> IO Int\n",
        "usesG = g\n",
    ));
}

#[test]
fn as_pattern_let_in_expression() {
    check_ok(concat!(
        "module M where\n",
        "data K = K Int\n",
        "g :: Int -> Int\n",
        "g x = let key@(K k) = K x in k\n",
        "usesG :: Int -> Int\n",
        "usesG = g\n",
    ));
}

#[test]
fn as_pattern_binding_var_subpattern() {
    // `name@subvar` (as-pattern binding a variable), not just `name@(Con ..)`.
    check_ok(concat!(
        "module M where\n",
        "g :: Int -> Int\n",
        "g x = let a@b = x in a + b\n",
        "usesG :: Int -> Int\n",
        "usesG = g\n",
    ));
}

#[test]
fn as_pattern_let_in_guard_do_then_trailing_guard() {
    // The faithful RST `resolveReferences` shape: an as-pattern `let` inside a
    // guard's `do` block, followed by another guard whose RHS uses the function
    // parameter. The parameter must remain in scope in the trailing guard.
    check_ok(concat!(
        "module M where\n",
        "data K = K Int\n",
        "f :: Int -> Maybe Int -> IO Int\n",
        "f x a\n",
        "  | Just ref <- a = do\n",
        "      let key@(K k) = K ref\n",
        "      return x\n",
        "  | otherwise = return x\n",
        "f x _ = return x\n",
        "usesF :: Int -> Maybe Int -> IO Int\n",
        "usesF = f\n",
    ));
}
