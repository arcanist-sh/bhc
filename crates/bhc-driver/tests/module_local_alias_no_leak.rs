//! Regression: a module-LOCAL type alias must not leak into modules that don't
//! import it.
//!
//! `imported_aliases` for a module was collected from EVERY module in the
//! registry, not just the ones it imports. So `Text.Pandoc.Readers.Docx.Parse`'s
//! module-local `type Target = T.Text` clobbered pandoc-types' canonical
//! `type Target = (Text, Text)` in unrelated modules — `Text.Pandoc.Writers.RST`
//! saw `Target` collapse to `Text` and failed with `expected Text, found
//! (Text, Text)`. Fix: scope `imported_aliases` to the module's actual imports.

use bhc_driver::Compiler;
use camino::Utf8PathBuf;

#[test]
fn local_alias_in_unimported_module_does_not_clobber() {
    let dep_dir = tempfile::tempdir().unwrap();
    let app_dir = tempfile::tempdir().unwrap();

    // Canonical: exports `type Pair = (Int, Int)`.
    std::fs::write(
        dep_dir.path().join("Canonical.hs"),
        "module Canonical (Pair) where\n\
         type Pair = (Int, Int)\n",
    )
    .unwrap();

    // Rogue: an UNRELATED module with a module-local `type Pair = Int`. It must
    // not leak into a module that imports only Canonical.
    std::fs::write(
        dep_dir.path().join("Rogue.hs"),
        "module Rogue where\n\
         type Pair = Int\n\
         rogue :: Pair -> Int\n\
         rogue x = x\n",
    )
    .unwrap();

    // App: imports Canonical only; uses Pair as the (Int, Int) tuple.
    std::fs::write(
        app_dir.path().join("Main.hs"),
        "module Main (main) where\n\
         import Canonical (Pair)\n\
         fstOf :: Pair -> Int\n\
         fstOf (a, _) = a\n\
         main :: IO ()\n\
         main = print (fstOf (1, 2))\n",
    )
    .unwrap();

    let compiler = Compiler::with_defaults().unwrap();
    let app = Utf8PathBuf::from(app_dir.path().to_str().unwrap());
    let dep = Utf8PathBuf::from(dep_dir.path().to_str().unwrap());

    let resolved = compiler
        .check_with_discovery_with_deps(&[app], &[dep])
        .unwrap();
    let main = resolved
        .iter()
        .find(|(n, _)| n == "Main")
        .expect("Main reported");
    assert!(
        main.1.is_ok(),
        "Rogue's local `type Pair = Int` must not leak into Main (imports only Canonical): {:?}",
        main.1
    );
}
