//! Regression: a signature with TWO constraint contexts —
//! `C1 a => C2 a => T` (two `=>` arrows) — must keep BOTH constraints.
//!
//! The parser parsed the first context and then parsed the remainder as a plain
//! function type, so a second `=>` context was swallowed and its constraint
//! silently dropped from the scheme. `Text.Pandoc.Parsing.General`'s
//! `indentWith :: (Stream s m Char, UpdateSourcePos s Char) => HasReaderOptions
//! st => Int -> ParsecT s st m Text` then lost `HasReaderOptions st`, so
//! `getOption` (a method of that class) reported `expected (HasReaderOptions t),
//! found (Int -> t)`. The same shape drove much of the `ToMetaValue` cluster in
//! JATS/LaTeX.Parsing. Nested contexts are now flattened into the scheme.

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
        "expected a two-context signature to retain both constraints, got {result:?}"
    );
}

#[test]
fn second_constraint_context_is_kept() {
    // `m x` needs the `C a` constraint, which sits in the SECOND context.
    // Pre-fix it was dropped and this failed with `expected (C t), found (t -> t)`.
    check_ok(concat!(
        "module M where\n",
        "class C a where { m :: a -> Int }\n",
        "f :: Show a => C a => a -> Int\n",
        "f x = m x\n",
    ));
}

#[test]
fn combined_context_still_works() {
    // The single-context form must keep working identically.
    check_ok(concat!(
        "module M where\n",
        "class C a where { m :: a -> Int }\n",
        "f :: (Show a, C a) => a -> Int\n",
        "f x = m x\n",
    ));
}
