//! Regression: `Data.List.NonEmpty`'s `:|` must have fixity `infixr 5` (like
//! `:`), not the default `infixl 9`.
//!
//! `get_operator_info` (bhc-parser) had no entry for `:|`, so it fell back to the
//! default `(9, Assoc::Left)`. That binds tighter than `:` (precedence 5), so
//! `x :| y : z` mis-parsed as `(x :| y) : z` instead of `x :| (y : z)`. The
//! second operand of `:|` was then forced to a list, e.g. `maximum (n :| m : ms)`
//! demanded `m :: [Int]` and produced `No instance for Num [Int]`. This is how
//! `Text.Pandoc.Writers.Muse`/`Writers.Shared`'s `gridTable`
//! (`maximum (length aligns :| length widths : map length (headers:rows))`)
//! failed.

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
        "expected `x :| y : z` to parse as `x :| (y : z)`, got {result:?}"
    );
}

#[test]
fn nonempty_cons_binds_looser_than_list_cons() {
    // `1 :| 2 : [3,4]` must be `1 :| (2 : [3,4])`. Before the fix it parsed as
    // `(1 :| 2) : [3,4]`, forcing `2 :: [Int]` -> `No instance for Num [Int]`.
    check_ok(concat!(
        "module M where\n",
        "import Data.List.NonEmpty (NonEmpty(..))\n",
        "g :: Int\n",
        "g = maximum (1 :| 2 : [3,4])\n",
    ));
}
