//! Regression: a function with an explicit type signature may use polymorphic
//! recursion (call itself at a different instantiation of its type).
//!
//! In a recursive binding group, typeck pre-registered each signatured def with
//! `Scheme::mono(sig.ty)` — the signature type but monomorphic — so every
//! recursive call shared the signature's type variables with the body. For a
//! self-referential result type this produced a spurious `infinite type`
//! (occurs check): e.g. `Text.Pandoc.Readers.Muse`'s `bulletListItemsUntil`,
//! whose result flows back through `listItemContentsUntil` into a recursive
//! call. An explicit signature licenses polymorphic recursion, so the def is now
//! registered with its full polymorphic scheme and recursive calls instantiate
//! fresh variables.

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
        "expected polymorphic recursion (with a signature) to type-check, got {result:?}"
    );
}

#[test]
fn signatured_function_allows_polymorphic_recursion() {
    // `size` recurses at `Nested [a]`, not `Nested a`. Its signature licenses
    // that; before the fix it failed with `infinite type` / occurs check.
    check_ok(concat!(
        "module M where\n",
        "data Nested a = Nil | Cons a (Nested [a])\n",
        "size :: Nested a -> Int\n",
        "size Nil = 0\n",
        "size (Cons _ xs) = 1 + size xs\n",
    ));
}
