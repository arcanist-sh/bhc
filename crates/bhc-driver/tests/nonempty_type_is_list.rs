//! Regression: the `NonEmpty a` type must unify with list values.
//!
//! BHC models `Data.List.NonEmpty`'s `:|` constructor and its operations as
//! list-valued (an approximation — see bhc-typeck context.rs `":|"`). But the
//! *type* `NonEmpty a` was a distinct `Con("NonEmpty")`, so a value built with
//! `:|` (a list) clashed with any `NonEmpty`-typed context:
//! `expected NonEmpty, found []`. This broke e.g. `Text.Pandoc.Writers.LaTeX.Table`,
//! whose `multicolumnDescriptor` takes a `NonEmpty ColWidth` fed from
//! `NonEmpty.map snd specs`.
//!
//! `NonEmpty a` is now registered as the type alias `[a]`, keeping the type and
//! the (list-valued) operations coherent.

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
        "expected `NonEmpty a` to accept a `:|`-built value, got {result:?}"
    );
}

#[test]
fn nonempty_type_accepts_cons_value() {
    // `1 :| [2,3]` is list-valued in BHC; a `NonEmpty Int` annotation must accept
    // it. Before the alias this failed with `expected NonEmpty, found []`.
    check_ok(concat!(
        "module M where\n",
        "import Data.List.NonEmpty (NonEmpty(..))\n",
        "g :: NonEmpty Int\n",
        "g = 1 :| [2, 3]\n",
        "-- also a NonEmpty-typed parameter must accept a list-shaped value\n",
        "h :: NonEmpty Int -> Int\n",
        "h xs = sum (toListLike xs)\n",
        "  where toListLike ys = ys\n",
    ));
}
