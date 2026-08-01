//! Regression: `where` clauses nested three or more levels deep must scope
//! their bindings. The lowering handled `where`-in-`where` inline to a fixed
//! depth of two — a `where`-binding whose RHS had its own `where`
//! (`a = b where b = c where c = ...`) had that innermost `where` discarded, so
//! the depth-3 binding came out `unbound variable`. The nested-where lowering
//! is now fully recursive (`lower_clause_rhs_with_wheres`).
//!
//! Text.Pandoc.Writers.ICML's `parStylesToDoc` has `indent = [.. indt ..] where
//! .. indt = ..` nested inside further `where`/`let`, which dropped `indt`.

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
fn where_nested_depth_three() {
    check_ok(concat!(
        "module M where\n",
        "h :: Int -> Int\n",
        "h s = a\n",
        "  where a = b\n",
        "          where b = c\n",
        "                  where c = s + 1\n",
        "usesH :: Int -> Int\n",
        "usesH = h\n",
    ));
}

#[test]
fn where_nested_depth_four() {
    check_ok(concat!(
        "module M where\n",
        "h :: Int -> Int\n",
        "h s = a\n",
        "  where a = b\n",
        "          where b = c\n",
        "                  where c = d\n",
        "                          where d = s + 1\n",
        "usesH :: Int -> Int\n",
        "usesH = h\n",
    ));
}

#[test]
fn where_nested_depth_three_with_params() {
    // A deep where-binding that takes parameters (lowered to a lambda) must
    // still thread its own where.
    check_ok(concat!(
        "module M where\n",
        "h :: Int -> Int\n",
        "h s = a s\n",
        "  where a n = b n\n",
        "          where b m = c m\n",
        "                  where c k = k + s\n",
        "usesH :: Int -> Int\n",
        "usesH = h\n",
    ));
}
