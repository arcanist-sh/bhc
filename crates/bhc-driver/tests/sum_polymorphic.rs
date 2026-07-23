//! Regression: `sum`/`product` must be `Num a => [a] -> a`, not pinned to Int.
//!
//! The builtin scheme was `[Int] -> Int`, so `sum [2.0, 3.0] :: Double` failed
//! with "expected Double, found Int" (and `1 - sum (map getColWidth …)` broke
//! `Text.Pandoc.Readers.HTML.Table`). The scheme is now polymorphic.

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
        "expected sum-over-Double program to type-check, got {result:?}"
    );
}

#[test]
fn sum_over_double_typechecks() {
    check_ok(concat!(
        "module M where\n",
        "h :: Double\n",
        "h = sum [2.0, 3.0]\n",
    ));
}

#[test]
fn product_over_double_typechecks() {
    check_ok(concat!(
        "module M where\n",
        "h :: Double -> Double\n",
        "h x = 1 - product [x, x]\n",
    ));
}
