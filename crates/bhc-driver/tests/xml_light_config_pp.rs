//! Regression: the `Text.XML.Light` stub must re-export the config
//! pretty-printer from `Text.XML.Light.Output` (`ppcElement`, `ppcTopElement`,
//! `ppcContent`, `defaultConfigPP`, `useShortEmptyTags`). They were present in
//! the `Text.XML.Light.Output` stub list but missing from `Text.XML.Light`
//! itself, so modules that `import qualified Text.XML.Light as Xml` and call
//! `Xml.ppcElement` saw `unbound variable`
//! (Text.Pandoc.Writers.DocBook, Text.Pandoc.Writers.ODT).

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
fn xml_light_config_pretty_printer_resolves() {
    check_ok(concat!(
        "module M where\n",
        "import qualified Text.XML.Light as Xml\n",
        "render :: Xml.Element -> String\n",
        "render el = Xml.ppcElement cfg el\n",
        "  where cfg = Xml.useShortEmptyTags (\\_ _ -> False) Xml.defaultConfigPP\n",
    ));
}
