//! Regression: citeproc's `Citation` record must survive a same-named REAL
//! constructor in scope.
//!
//! The generic stub-module indirection resolved `Citeproc.Citation` through the
//! unqualified name `Citation` — with Text.Pandoc.Definition (or any module
//! exporting a real `Citation`) imported, record construction bound to that
//! unrelated constructor and Readers.EndNote failed with `expected
//! (Citeproc.Citation Text), found Citation`. The colliding export now gets a
//! dedicated stub (like the Djot modules), with a curated record scheme
//! (`citationId -> citationNoteNumber -> citationItems -> Citeproc.Citation a`).
//! Also covers the `CitationItemType` constructors (`Citeproc.NormalCite`) that
//! EndNote sets on its citation items.

use bhc_driver::Compiler;
use camino::Utf8PathBuf;

#[test]
fn citeproc_citation_record_beats_same_named_real_constructor() {
    let dep_dir = tempfile::tempdir().unwrap();
    let app_dir = tempfile::tempdir().unwrap();

    // A real module exporting an unrelated `Citation` constructor (stands in
    // for pandoc-types' Text.Pandoc.Definition).
    std::fs::write(
        dep_dir.path().join("Defn.hs"),
        "module Defn (Citation (..)) where\n\
         data Citation = Citation\n\
           { citationId :: String\n\
           , citationHash :: Int\n\
           }\n",
    )
    .unwrap();

    // Mirrors Readers.EndNote: both the real module and qualified Citeproc in
    // scope; the record construction must resolve to the citeproc stub.
    std::fs::write(
        app_dir.path().join("Main.hs"),
        "module Main (main) where\n\
         import Defn\n\
         import Data.Text (Text)\n\
         import qualified Citeproc\n\
         mkCite :: [Citeproc.CitationItem Text] -> Citeproc.Citation Text\n\
         mkCite items = Citeproc.Citation{\n\
                              Citeproc.citationId = Nothing\n\
                            , Citeproc.citationNoteNumber = Nothing\n\
                            , Citeproc.citationItems = items\n\
                            }\n\
         normal :: Citeproc.CitationItemType\n\
         normal = Citeproc.NormalCite\n\
         main :: IO ()\n\
         main = pure ()\n",
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
        "`Citeproc.Citation{{..}}` must build the citeproc stub, not Defn's \
         Citation: {:?}",
        main.1
    );
}
