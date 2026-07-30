//! Regression: `traverse` (and `for`, `sequenceA`) are polymorphic in the
//! applicative — `traverse :: Applicative f => (a -> f b) -> t a -> f (t b)` —
//! not pinned to `IO`.
//!
//! The `Data.Traversable` builtins were schemed `(a -> IO b) -> c -> IO d`,
//! hardcoding `f = IO`. Any non-IO use then forced `IO ~ <that functor>`, e.g.
//! `Text.Pandoc.Writers.Shared`'s `traverse toSubscriptInline` (where
//! `toSubscriptInline :: Inline -> Maybe Inline`) failed with
//! `expected Maybe, found IO`. The functor is now a quantified var of kind
//! `* -> *`.

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
        "expected `traverse` with a Maybe-valued function to type-check, got {result:?}"
    );
}

#[test]
fn traverse_works_in_maybe() {
    // `traverse Just :: [a] -> Maybe [a]`. Pre-fix `traverse` was IO-pinned and
    // this failed with `Maybe ~ IO`.
    check_ok(concat!(
        "module M where\n",
        "g :: [Int] -> Maybe [Int]\n",
        "g = traverse Just\n",
    ));
}
