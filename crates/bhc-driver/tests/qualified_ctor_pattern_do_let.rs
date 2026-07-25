//! Regression: a *qualified* constructor in a parenthesized pattern binding
//! must be recognized as a pattern (not mis-parsed), especially inside a
//! do-block `let`.
//!
//! `parse_value_decl_with_doc`'s `is_pattern_start` check (bhc-parser) — used to
//! decide whether `( … ) = e` is a pattern binding — matched `ConId` but not
//! `QualConId`. So `let (D.C a b c) = w` (a qualified constructor pattern) failed
//! the check, fell through to a mis-parse, and inside a do-block it COLLAPSED the
//! do-layout: the pattern's field bindings were dropped and every later use
//! (`a`/`b`/`c`) reported `unbound variable`. A single-let degraded to a working
//! `let … in`, but two lets collapsed into an infix expression.
//!
//! This is the root cause of the large lowering-error cascades in
//! `Writers.LaTeX.Table` / `Writers.Docx.Table` / `Writers.OpenDocument`, whose
//! table code destructures `Ann.Cell`/`Cell` via `let (Ann.Cell …) = …` followed
//! by more `let`s.

use bhc_driver::Compiler;
use camino::Utf8PathBuf;

fn write(dir: &camino::Utf8Path, name: &str, src: &str) -> Utf8PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).expect("write module");
    path
}

#[test]
fn qualified_ctor_pattern_in_do_let_keeps_bindings() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = Utf8PathBuf::from_path_buf(tmp.path().to_path_buf()).expect("utf8 dir");

    let def = write(
        &dir,
        "Def.hs",
        "module Def (C(..)) where\ndata C = C Int Int Int\n",
    );
    // `let (D.C a b c) = w` uses a QUALIFIED constructor pattern, followed by a
    // second `let` that references its fields — the exact shape that mis-parsed.
    let use_it = write(
        &dir,
        "Use.hs",
        concat!(
            "module Use where\n",
            "import qualified Def as D\n",
            "f :: D.C -> IO Int\n",
            "f w = do\n",
            "  let (D.C a b c) = w\n",
            "  let x = a + b + c\n",
            "  return x\n",
        ),
    );

    let compiler = Compiler::with_defaults().expect("compiler");
    let results = compiler.check_files_ordered(&[def, use_it]).expect("check");

    let use_result = results.iter().find(|(name, _)| name.contains("Use"));
    assert!(
        matches!(use_result, Some((_, Ok(())))),
        "Use should type-check (qualified ctor pattern must not collapse the do-block); results: {results:?}"
    );
}
