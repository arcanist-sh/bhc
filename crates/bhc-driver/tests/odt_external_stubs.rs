//! Regression: three external-package stubs that Text.Pandoc.Writers.ODT needs
//! were missing from their stub export lists — `parseRelativeReference`
//! (Network.URI, which had `parseURIReference`/`isRelativeReference` but not
//! this one), `zEntries` (Codec.Archive.Zip archive accessor), and `fromColor`
//! (Skylighting `Color` conversion). Importers saw them as `unbound variable`.

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
fn odt_external_package_names_resolve() {
    check_ok(concat!(
        "module M where\n",
        "import Network.URI (parseRelativeReference)\n",
        "import Codec.Archive.Zip (zEntries, Archive)\n",
        "import Skylighting (fromColor, Color)\n",
        "relPath :: Maybe a\n",
        "relPath = parseRelativeReference \"images/logo.png\"\n",
        "entryCount :: Archive -> Int\n",
        "entryCount ar = length (zEntries ar)\n",
        "colorString :: Color -> String\n",
        "colorString = fromColor\n",
    ));
}

#[test]
fn xml_light_element_conversions_resolve() {
    // Text.Pandoc.XML.Light's `fromXLElement`/`toXLElement` (used by
    // Powerpoint.Output and Docx.OpenXML) were missing from that stub list.
    check_ok(concat!(
        "module M where\n",
        "import Text.Pandoc.XML.Light (fromXLElement, Element)\n",
        "convert :: a -> Element\n",
        "convert = fromXLElement\n",
    ));
}
