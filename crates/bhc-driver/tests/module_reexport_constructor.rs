//! Regression: a `module X` re-export must propagate X's constructors from X
//! *specifically*, so a same-named constructor in another module can't shadow
//! them.
//!
//! `Text.Pandoc.Translations` re-exports `module Text.Pandoc.Translations.Types`,
//! whose nullary `Term` constructor `Figure` was shadowed in the driver's
//! registry merge by `Text.Pandoc.Definition.Figure :: … -> Block` (picked by a
//! name-only scan across all modules) — so `doTerm Translations.Figure` in
//! `Readers.LaTeX.Inline` inferred `Figure` as a 3-arg function to `Block`.

use bhc_driver::Compiler;
use camino::Utf8PathBuf;

fn write(dir: &camino::Utf8Path, name: &str, src: &str) -> Utf8PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).expect("write module");
    path
}

#[test]
fn module_reexport_picks_right_constructor() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let dir = Utf8PathBuf::from_path_buf(tmp.path().to_path_buf()).expect("utf8 dir");

    // `MyCon` is nullary here (type `Term`) …
    let term = write(
        &dir,
        "TermTypes.hs",
        "module TermTypes (Term(..)) where\ndata Term = MyCon | TermOnly\n",
    );
    // … and a 2-arg constructor of a different type there. Both land in the
    // registry, creating the same-name ambiguity.
    let other = write(
        &dir,
        "OtherMod.hs",
        "module OtherMod (Blk(..)) where\ndata Blk = MyCon Int Int | BlkOnly\n",
    );
    let reexp = write(
        &dir,
        "Reexp.hs",
        "module Reexp (module TermTypes) where\nimport TermTypes\n",
    );
    let use_it = write(
        &dir,
        "UseIt.hs",
        "module UseIt where\nimport qualified Reexp as T\nf :: T.Term\nf = T.MyCon\n",
    );

    let compiler = Compiler::with_defaults().expect("compiler");
    let results = compiler
        .check_files_ordered(&[term, other, reexp, use_it])
        .expect("check");

    // `T.MyCon` must resolve to `TermTypes.MyCon :: Term`, not
    // `OtherMod.MyCon :: Int -> Int -> Blk`.
    let use_result = results.iter().find(|(name, _)| name.contains("UseIt"));
    assert!(
        matches!(use_result, Some((_, Ok(())))),
        "UseIt should type-check; results: {results:?}"
    );
}
