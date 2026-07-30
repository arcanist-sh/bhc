//! Regression: `Text.XML.Light`'s `elChildren` returns `[Element]`, not
//! `[Content]`.
//!
//! bhc bundled `elChildren` into the same scheme arm as `elContent`, typing
//! both `Element -> [Content]`. But `elChildren = onlyElems . elContent` keeps
//! only the child *Elements*. Code like `Text.Pandoc.Readers.FB2`'s
//! `mapM parseChild (elChildren e)` — where `parseChild :: Element -> m a` —
//! then failed with `expected Element, found Content` (12 such errors in FB2
//! alone, plus most of Docx.Parse). `elChildren` is now typed
//! `Element -> [Element]`; `elContent` keeps `Element -> [Content]`.

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
        "expected `elChildren` to yield `[Element]` (usable where Element is \
         expected), got {result:?}"
    );
}

#[test]
fn elchildren_yields_elements_not_content() {
    // `elName :: Element -> QName`, so `map elName (elChildren e)` only checks
    // if `elChildren e :: [Element]`. Pre-fix it was `[Content]` and this
    // failed with `expected Element, found Content`.
    check_ok(concat!(
        "module M where\n",
        "import Text.Pandoc.XML.Light (elChildren, elName)\n",
        "h e = map elName (elChildren e)\n",
    ));
}
