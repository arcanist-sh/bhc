//! Regression coverage for the fixes that completed the Pandoc grind at
//! 221/221 (2026-08-04): declared fixities, TH expression splices, and the
//! last scheme/stub gaps.

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
        "expected the module to check"
    );
}

#[test]
fn declared_fixity_wins_over_default() {
    // Readers.Org.Meta's `infix 0 ~~>`: without the declaration the operator
    // defaults to infixl 9 and `"k" ~~> p `andThen` f` mis-groups as
    // `("k" ~~> p) `andThen` f`, feeding the handler a TUPLE — all 84 of
    // Org.Meta's errors were this one mis-grouping.
    check_ok(concat!(
        "module M where\n",
        "import qualified Data.Map as Map\n",
        "infix 0 ~~>\n",
        "(~~>) :: a -> b -> (a, b)\n",
        "a ~~> b = (a, b)\n",
        "andThen :: Maybe Int -> Int -> Maybe Int\n",
        "andThen m n = fmap (+ n) m\n",
        "handlers :: Map.Map String (Maybe Int)\n",
        "handlers = Map.fromList\n",
        "  [ \"a\" ~~> Just 1 `andThen` 2\n",
        "  , \"b\" ~~> Nothing `andThen` 3\n",
        "  ]\n",
    ));
}

#[test]
fn th_expression_splice_is_permissive() {
    // `$(embedFile \"f\")` (Data.BakedIn / Citeproc.Data / Readers.DocBook):
    // a prefix-position `$(` parses as a splice with a fresh result type.
    check_ok(concat!(
        "module M where\n",
        "import Data.FileEmbed (embedFile)\n",
        "import Data.ByteString (ByteString)\n",
        "manual :: ByteString\n",
        "manual = $(embedFile \"MANUAL.txt\")\n",
        "pair :: (FilePath, ByteString)\n",
        "pair = (\"MANUAL.txt\", $(embedFile \"MANUAL.txt\"))\n",
    ));
}

#[test]
fn decode_utf8_with_takes_handler_first() {
    // `decodeUtf8With lenientDecode bs` (Text.Pandoc.App): the curated scheme
    // was unary (`ByteString -> Text`), so the second application failed.
    check_ok(concat!(
        "module M where\n",
        "import Data.Text (Text)\n",
        "import qualified Data.Text.Encoding as TE\n",
        "import qualified Data.Text.Encoding.Error as TE\n",
        "import Data.ByteString (ByteString)\n",
        "decode :: ByteString -> Text\n",
        "decode = TE.decodeUtf8With TE.lenientDecode\n",
    ));
}

#[test]
fn unregistered_assoc_family_unifies_permissively() {
    // Readers.Docx: `getStyleName :: a -> StyleName a` — an imported class
    // method whose associated family instance lives in another module. The
    // `StyleName t` application must not hard-fail against the concrete
    // `ParaStyleName` the caller supplies.
    let dep_dir = tempfile::tempdir().unwrap();
    let app_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dep_dir.path().join("Styles.hs"),
        "module Styles (HasStyleName, getStyleName, ParaStyleName (..), ParStyle (..)) where\n\
         newtype ParaStyleName = ParaStyleName String\n\
         data ParStyle = ParStyle ParaStyleName\n\
         class HasStyleName a where\n\
           type StyleName a\n\
           getStyleName :: a -> StyleName a\n\
         instance HasStyleName ParStyle where\n\
           type StyleName ParStyle = ParaStyleName\n\
           getStyleName (ParStyle n) = n\n",
    )
    .unwrap();
    std::fs::write(
        app_dir.path().join("Main.hs"),
        "module Main (main) where\n\
         import Styles\n\
         nameOf :: ParStyle -> ParaStyleName\n\
         nameOf = getStyleName\n\
         main :: IO ()\n\
         main = pure ()\n",
    )
    .unwrap();
    let compiler = Compiler::with_defaults().unwrap();
    let app = camino::Utf8PathBuf::from(app_dir.path().to_str().unwrap());
    let dep = camino::Utf8PathBuf::from(dep_dir.path().to_str().unwrap());
    let resolved = compiler
        .check_with_discovery_with_deps(&[app], &[dep])
        .unwrap();
    let main = resolved
        .iter()
        .find(|(n, _)| n == "Main")
        .expect("Main reported");
    assert!(
        main.1.is_ok(),
        "cross-module associated family must unify permissively: {:?}",
        main.1
    );
}
