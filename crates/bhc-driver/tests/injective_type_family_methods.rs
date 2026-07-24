//! Regression: an injectivity annotation on a class's associated type family
//! must not drop the methods declared after it.
//!
//! `parse_assoc_type_decl` parsed `type Token x = a` as a defaulted associated
//! type but left the trailing injectivity clause `| a -> x` unconsumed. The
//! class-body loop then saw `|` where it expected a `;`, stopped early, and
//! silently dropped every method after the type family. Typeck limped along but
//! lowering reported the methods as `unbound variable` — this is exactly how
//! `Text.Pandoc.Readers.Roff.Escape` failed (its `RoffLikeLexer` class has
//! `type State x = a | a -> x` / `type Token x = a | a -> x` followed by `emit`,
//! `backslash`, `escString`, …), cascading to Roff/Man/Mdoc being skipped.
//!
//! The parser now consumes and discards the injectivity annotation, so the
//! methods survive and lower correctly.

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
        "expected injective-ATF class program to check + lower, got {result:?}"
    );
}

#[test]
fn injective_associated_type_family_keeps_class_methods() {
    // `escString` (declared after the injective family) must remain a resolvable
    // class method — `escape`'s body uses it through the `C x` constraint.
    check_ok(concat!(
        "{-# LANGUAGE TypeFamilies #-}\n",
        "module M (escape) where\n",
        "type Lexer m x a = m a\n",
        "class C x where\n",
        "  type State x = a | a -> x\n",
        "  type Token x = a | a -> x\n",
        "  backslash :: Monad m => Lexer m x ()\n",
        "  escString :: Monad m => Lexer m x (Token x)\n",
        "escape :: (Monad m, C x) => Lexer m x (Token x)\n",
        "escape = do { backslash; escString }\n",
    ));
}
