//! Regression: the `Skylighting` stub must export `defStyle` and the
//! `TokenStyle` record accessors (`tokenColor`, `tokenBackground`, `tokenBold`,
//! `tokenItalic`, `tokenUnderline`). They were missing from the stub export
//! list, so importers were left with `unbound variable: defStyle` /
//! `tokenBold` / ... (Text.Pandoc.Writers.Man and
//! Text.Pandoc.Writers.Powerpoint.Presentation both use them to build styling
//! from a highlighting theme).

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
fn skylighting_defstyle_and_tokenstyle_fields_resolve() {
    check_ok(concat!(
        "module M where\n",
        "import Skylighting (defStyle, TokenStyle(..))\n",
        "fonts :: [Char]\n",
        "fonts = ['B' | tokenBold defStyle]\n",
        "     ++ ['I' | tokenItalic defStyle || tokenUnderline defStyle]\n",
    ));
}
