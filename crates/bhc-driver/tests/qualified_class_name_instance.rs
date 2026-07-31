//! Regression: an instance head with a QUALIFIED class name must parse.
//!
//! `parse_conid` (used for the instance's class name) accepted only an
//! unqualified `ConId`, so `instance Cat.Category (ArrowState s) where …`
//! failed to parse. Error recovery then reparsed the instance's `where`
//! methods as TOP-LEVEL bindings — so an infix method like `(.)` became a
//! top-level operator definition that SHADOWED the builtin `.` (function
//! composition) at every use site in the module. In
//! `Text.Pandoc.Readers.ODT.Arrows.State` this made the constructors
//! (`ArrowState . first`, etc.) resolve `.` to the ArrowState Category
//! composition, yielding 15 `expected ArrowState, found ->` errors.

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
        "instance with a qualified class name should parse and not leak its methods"
    );
}

#[test]
fn qualified_class_name_instance_with_infix_operator_method() {
    // The `.` method of the qualified-class instance must NOT leak to the top
    // level; `modifyState`'s `.` must stay the builtin function composition.
    check_ok(concat!(
        "{-# LANGUAGE TupleSections #-}\n",
        "module M where\n",
        "import Control.Arrow\n",
        "import qualified Control.Category as Cat\n",
        "newtype ArrowState state a b = ArrowState\n",
        "  { runArrowState :: (state, a) -> (state, b) }\n",
        "modifyState :: (state -> state) -> ArrowState state a a\n",
        "modifyState = ArrowState . first\n",
        "instance Cat.Category (ArrowState s) where\n",
        "  id = ArrowState Cat.id\n",
        "  arrow2 . arrow1 = ArrowState $ runArrowState arrow2 . runArrowState arrow1\n",
    ));
}
