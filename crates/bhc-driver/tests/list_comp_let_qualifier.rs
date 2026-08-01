//! Regression: a `let` qualifier in a list comprehension must scope its
//! bindings over the head expression and the following qualifiers.
//!
//! Text.Pandoc.Readers.RST's `mkAttr` has
//!   fields' = [(k, v') | (k, v) <- fields, let v' = trimr v, k /= "name"]
//! Before the fix the comprehension body was lowered *before* the `let`
//! qualifier's names were bound, so `v'` came out `unbound variable: v'`
//! (the do-notation `let` scoped correctly, but the list-comprehension one
//! did not).

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
fn let_qualifier_scopes_into_head() {
    check_ok(concat!(
        "module M where\n",
        "f :: [(Int, Int)] -> [(Int, Int)]\n",
        "f fields = [(k, v') | (k, v) <- fields, let v' = v + 1, k /= 0]\n",
        "usesF :: [(Int, Int)] -> [(Int, Int)]\n",
        "usesF = f\n",
    ));
}

#[test]
fn let_qualifier_multiline_leading_comma() {
    // The faithful RST shape: multi-line comprehension, leading-comma
    // qualifiers, `let` bound var used in the head.
    check_ok(concat!(
        "module M where\n",
        "f :: [(Int, Int)] -> [(Int, Int)]\n",
        "f fields = fields'\n",
        "  where fields' = [(k, v') | (k, v) <- fields\n",
        "                           , let v' = v + 1\n",
        "                           , k /= 0, k /= 1]\n",
        "usesF :: [(Int, Int)] -> [(Int, Int)]\n",
        "usesF = f\n",
    ));
}

#[test]
fn let_qualifier_scopes_into_later_guard() {
    // A `let`-bound name must also be visible to qualifiers that follow it.
    check_ok(concat!(
        "module M where\n",
        "f :: [Int] -> [Int]\n",
        "f xs = [y | x <- xs, let y = x * 2, y > 3]\n",
        "usesF :: [Int] -> [Int]\n",
        "usesF = f\n",
    ));
}
