//! Regression: Text.Pandoc.Writers.CslJson imports `Lang`/`parseLang` from
//! `Citeproc` and `NumberFormat(Generic)` from `Data.Aeson.Encode.Pretty`.
//! The `Citeproc` stub list had `Reference`/`Locale` but not `Lang`/`parseLang`,
//! and the `Data.Aeson.Encode.Pretty` list had `Config`/`Indent` but not
//! `NumberFormat`/`Generic` — so both came out `unbound`.

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
fn citeproc_lang_and_aeson_pretty_numberformat_resolve() {
    check_ok(concat!(
        "module M where\n",
        "import Citeproc (parseLang, Lang(..))\n",
        "import Data.Aeson.Encode.Pretty (Config(..), NumberFormat(Generic), defConfig)\n",
        "defaultLang :: Lang\n",
        "defaultLang = Lang \"en\" Nothing (Just \"US\") [] [] []\n",
        "cfg :: Config\n",
        "cfg = defConfig { confNumFormat = Generic }\n",
        "readLang :: String -> Maybe Lang\n",
        "readLang s = either (const Nothing) Just (parseLang s)\n",
    ));
}
