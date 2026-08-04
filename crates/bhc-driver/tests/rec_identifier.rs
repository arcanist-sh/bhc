//! Regression: `rec` is a valid Haskell 2010 identifier, not a reserved word.
//! It is only special inside RecursiveDo/Arrows `rec` blocks (statement
//! position). The lexer used to map `rec` unconditionally to a keyword token,
//! so `addId rec = ...` in Text.Pandoc.Readers.RIS failed to parse and the whole
//! binding was silently dropped (surfacing as a bogus `unbound variable` at the
//! use site).

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
fn rec_as_function_name() {
    check_ok(concat!(
        "module M where\n",
        "rec :: Int -> Int\n",
        "rec n = n + 1\n",
    ));
}

#[test]
fn rec_as_parameter_in_where_binding() {
    // Mirrors RIS's `addId rec = if ... then rec{...} else rec` shape: a
    // parametered where-binding whose parameter is named `rec`, followed by
    // sibling bindings. The whole outer binding must not be dropped.
    check_ok(concat!(
        "module M where\n",
        "f :: Int -> Int\n",
        "f x = addId x\n",
        "  where\n",
        "   addId rec = if rec == 0 then 1 else rec\n",
        "   unused = 5\n",
    ));
}
