//! Regression: djot (`Djot.AST`) is stubbed, and its exports must NOT be
//! conflated with same-named builtins, and its `Node` is polymorphic.
//!
//! - `Djot.AST.div` was resolving to the integer `div` (`a -> a -> a`), so
//!   `fmap f . D.div` (Text.Pandoc.Writers.Djot) saw a 2-arg function where a
//!   functor was expected. djot exports now get dedicated stubs.
//! - `D.Node` (`Node a`) was monomorphic, so `convertBlock`'s
//!   `D.Node _ _ _ :: D.Node D.Block` and `convertInline`'s `D.Node D.Inline`
//!   (Text.Pandoc.Readers.Djot) shared the parameter (`expected D.Block, found
//!   D.Inline`). It now has a polymorphic `Pos -> Attr -> a -> Node a` scheme.

use bhc_driver::Compiler;
use std::io::Write;

fn check_ok(source: &str) {
    let mut file = tempfile::Builder::new().suffix(".hs").tempfile().expect("temp");
    file.write_all(source.as_bytes()).expect("write");
    let path = camino::Utf8Path::from_path(file.path()).expect("utf8");
    let compiler = Compiler::with_defaults().expect("compiler");
    assert!(
        compiler.check_file(path).is_ok(),
        "expected djot stub usage to type-check"
    );
}

#[test]
fn djot_div_is_not_integer_div() {
    // `D.div` is a block wrapper, not integer division, so `fmap g . D.div`
    // (D.div as a 1-arg function producing a functor) must check.
    check_ok(concat!(
        "module M where\n",
        "import qualified Djot.AST as D\n",
        "f x = fmap D.str . D.div <$> x\n",
    ));
}

#[test]
fn djot_node_is_polymorphic_in_content() {
    // `D.Node` used at two different content types in two functions.
    check_ok(concat!(
        "module M where\n",
        "import qualified Djot.AST as D\n",
        "f :: D.Node D.Block -> Int\n",
        "f (D.Node p a bl) = 0\n",
        "g :: D.Node D.Inline -> Int\n",
        "g (D.Node p a il) = 1\n",
    ));
}
