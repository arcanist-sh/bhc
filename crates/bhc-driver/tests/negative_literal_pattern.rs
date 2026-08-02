//! Regression: negative-literal patterns (`-1`, `-2.5`) must parse. The lexer
//! emits a `Minus` token followed by the numeric literal, but the atom-pattern
//! parser had no `Minus` case, so a `-1 ->` case alternative was a parse error.
//! Error recovery usually contained it, but after a multi-line type signature
//! it dropped the whole enclosing function (surfacing as "unbound variable").
//!
//! Text.Pandoc.Writers.ConTeXt's `sectionLevelToText` has a multi-line
//! signature and `case hdrLevel + shift of { -1 -> literal "part"; ... }`,
//! which dropped the function.

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
fn negative_int_case_pattern() {
    check_ok(concat!(
        "module M where\n",
        "f :: Int -> Int\n",
        "f x = case x of\n",
        "        -1 -> 100\n",
        "        _  -> 0\n",
        "usesF :: Int -> Int\n",
        "usesF = f\n",
    ));
}

#[test]
fn negative_pattern_after_multiline_signature() {
    // The ConTeXt shape: a multi-line signature followed by a `-1` case pattern
    // must not drop the function.
    check_ok(concat!(
        "module M where\n",
        "g :: Int\n",
        "  -> Int\n",
        "g hdrLevel = case hdrLevel of\n",
        "               -1 -> 100\n",
        "               _  -> 0\n",
        "usesG :: Int -> Int\n",
        "usesG = g\n",
    ));
}

#[test]
fn negative_float_and_multiple_negative_alts() {
    check_ok(concat!(
        "module M where\n",
        "f :: Double -> Int\n",
        "f x = case x of\n",
        "        -1.5 -> 1\n",
        "        -2.0 -> 2\n",
        "        _    -> 0\n",
        "usesF :: Double -> Int\n",
        "usesF = f\n",
    ));
}
