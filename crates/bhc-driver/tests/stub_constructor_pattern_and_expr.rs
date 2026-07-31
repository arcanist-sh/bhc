//! Regression: an arity-less STUB constructor used as both a pattern and an
//! expression in one module must not conflict with itself.
//!
//! A stub constructor from a bhc-stubbed library module (here
//! `Text.Pandoc.XML.Light`'s `Content` constructor `Text`) has no arity info,
//! so `register_lowered_builtins` fell back to `Scheme::mono(fresh)` — a single
//! FREE (unquantified) type variable SHARED across every use. Using such a
//! constructor as a pattern (`Text (CData _ s _)`, forcing arity 1) and as an
//! expression (`Text (CData …)`) then pinned that one var two different ways
//! and failed with `expected (t -> t), found Content`. This blocked
//! `Text.Pandoc.Readers.JATS` (`parseBlock`/`parseInline`/`elementToStr`).
//!
//! The fallback now yields a properly polymorphic `forall a. a`, so each use
//! instantiates its own fresh variable.

use bhc_driver::Compiler;
use std::io::Write;

#[test]
fn stub_constructor_used_as_pattern_and_expression() {
    let mut file = tempfile::Builder::new()
        .suffix(".hs")
        .tempfile()
        .expect("temp");
    file.write_all(
        b"module M where\n\
          import qualified Data.Text as T\n\
          import Text.Pandoc.XML.Light\n\
          elementToStr :: Content -> Content\n\
          elementToStr (Elem _) = Text (CData CDataText undefined Nothing)\n\
          elementToStr x = x\n\
          consumer :: Content -> Int\n\
          consumer (Elem _) = 0\n\
          consumer (CRef _) = 1\n\
          consumer (Text (CData _ s _)) = if T.null s then 2 else 3\n",
    )
    .expect("write");
    let path = camino::Utf8Path::from_path(file.path()).expect("utf8");

    let compiler = Compiler::with_defaults().expect("compiler");
    assert!(
        compiler.check_file(path).is_ok(),
        "a stub constructor used as both a pattern and an expression should check"
    );
}
