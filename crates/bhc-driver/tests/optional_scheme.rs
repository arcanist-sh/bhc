//! Regression: the Applicative/Parsec `optional` combinator must not be
//! mistyped as `[Char] -> Bool`.
//!
//! The builtin `optional` (a `DefKind::Value`) fell into a DefId-collision gap:
//! its lowering DefId happened to match a `[Char] -> Bool` op registered by
//! `bhc-typeck`'s `register_primitive_ops` (the lowering `builtin_funcs` list and
//! the typeck `ops` list have drifted out of order), and no
//! `register_lowered_builtins` arm covered the *non-stub* builtin `optional`, so
//! it kept that stale scheme. Every `p <* optional q` in a Parsec `do` then
//! type-checked with `expected (f a), found Bool` (e.g. `Text.Pandoc.CSS`).
//!
//! These check that programs using `optional` type-check.

use bhc_driver::Compiler;
use std::io::Write;

fn check_ok(source: &str) {
    let mut file = tempfile::Builder::new()
        .suffix(".hs")
        .tempfile()
        .expect("create temp file");
    file.write_all(source.as_bytes()).expect("write source");
    let path = camino::Utf8Path::from_path(file.path()).expect("utf8 path");

    let compiler = Compiler::with_defaults().expect("compiler");
    let result = compiler.check_file(path);
    assert!(
        result.is_ok(),
        "expected `optional` program to type-check, got {result:?}"
    );
}

#[test]
fn optional_typechecks_in_applicative_context() {
    check_ok(
        "module M where\n\
         useOpt :: Maybe Int -> Maybe (Maybe Int)\n\
         useOpt x = optional x\n",
    );
}

#[test]
fn optional_left_seq_typechecks() {
    // `p <* optional q` — the exact shape that broke Text.Pandoc.CSS.
    check_ok(
        "module M where\n\
         both :: Maybe Int -> Maybe Int -> Maybe Int\n\
         both p q = p <* optional q\n",
    );
}
