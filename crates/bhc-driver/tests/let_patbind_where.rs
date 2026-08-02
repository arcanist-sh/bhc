//! Regression: a `let`-bound pattern binding with its own `where` clause
//! (`let (a, b) = e where f = ...`) must scope the `where` bindings. The
//! pattern-binding branch of the let desugarer lowered the RHS directly and
//! dropped the `where` (unlike the regular-binding branch, which threaded it),
//! so `f` came out `unbound variable`.
//!
//! Text.Pandoc.Writers.ICML's `parStylesToDoc` has
//!   let (isBulletList, isOrderedList) = findList ... where findList ... = ...
//! which dropped `findList`.

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
fn let_tuple_pattern_binding_with_where() {
    check_ok(concat!(
        "module M where\n",
        "makeStyle :: Int -> Bool\n",
        "makeStyle s =\n",
        "  let (a, b) = findList s\n",
        "        where findList 0 = (True, False)\n",
        "              findList n = (False, True)\n",
        "  in a && b\n",
        "usesM :: Int -> Bool\n",
        "usesM = makeStyle\n",
    ));
}

#[test]
fn let_tuple_pattern_binding_with_recursive_where() {
    check_ok(concat!(
        "module M where\n",
        "makeStyle :: Int -> Bool\n",
        "makeStyle s =\n",
        "  let (a, b) = findList s\n",
        "        where findList n = if n == 0 then (True, False) else findList (n - 1)\n",
        "  in a && b\n",
        "usesM :: Int -> Bool\n",
        "usesM = makeStyle\n",
    ));
}
