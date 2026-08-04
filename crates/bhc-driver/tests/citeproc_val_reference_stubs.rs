//! Regression: Text.Pandoc.Readers.RIS imports `Val(..)`, `Reference(..)`,
//! `ItemId(..)`, `Date(..)`, `DateParts(..)` from `Citeproc`. The base `Citeproc`
//! stub list re-exports these, so the `Val` constructors (`TextVal`, `FancyVal`,
//! `NamesVal`, `DateVal`), the `Reference` accessors (`referenceId`,
//! `referenceType`, `referenceVariables`) and `ItemId`/`unItemId` must all
//! resolve — including the constructors used as patterns.

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
fn citeproc_val_constructors_and_reference_accessors_resolve() {
    check_ok(concat!(
        "module M where\n",
        "import Citeproc (Reference(..), ItemId(..), Val(..))\n",
        "tagOf :: Val Text -> Int\n",
        "tagOf v = case v of\n",
        "  TextVal _ -> 0\n",
        "  FancyVal _ -> 1\n",
        "  NamesVal _ -> 2\n",
        "  DateVal _ -> 3\n",
        "idText :: Reference Text -> ItemId\n",
        "idText r = referenceId r\n",
        "wrap :: Text -> ItemId\n",
        "wrap = ItemId\n",
    ));
}
