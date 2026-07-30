//! Regression: a numeric literal in a *pattern* is overloaded (`Num a => a` /
//! `Fractional a => a`), exactly like a numeric literal expression — not a
//! monomorphic `Int`/`Float`.
//!
//! Pattern literals were typed as concrete `Int`, so matching `0` against a
//! `Double` failed: `Just 0` over `Maybe Double`, or
//! `case width of (0 :: Double) -> ..` in `Text.Pandoc.Readers.LaTeX`, gave
//! `expected Double, found Int`.

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
        "expected numeric literal patterns to be overloaded, got {result:?}"
    );
}

#[test]
fn numeric_literal_pattern_matches_double() {
    // `0` matched against Double (via annotation and via context), and against
    // Int, must all check.
    check_ok(concat!(
        "module M where\n",
        "f :: Maybe Double -> Int\n",
        "f (Just (0 :: Double)) = 1\n",
        "f (Just 0)             = 2\n",
        "f _                    = 0\n",
        "classify :: Double -> Int\n",
        "classify x = case x of { 0 -> 1; _ -> 2 }\n",
        "count :: Int -> Int\n",
        "count 0 = 0\n",
        "count n = n\n",
    ));
}
