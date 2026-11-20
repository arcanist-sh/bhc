//! Regression: `TL.Text` in a signature is lazy Text in the module's
//! interface.
//!
//! The qualifier was dropped and the type written as `Text`, so an importer
//! saw strict Text: pandoc's ANSI writer, `toStrict $ D.renderANSI …`, failed
//! with "expected LazyText, found Text".

use bhc_driver::CompilerBuilder;
use camino::Utf8PathBuf;

fn utf8(path: &std::path::Path) -> Utf8PathBuf {
    Utf8PathBuf::from(path.to_str().unwrap())
}

#[test]
fn qualified_lazy_text_survives_the_interface() {
    let src = tempfile::tempdir().unwrap();
    let odir = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();

    std::fs::write(
        src.path().join("Producer.hs"),
        concat!(
            "module Producer (lazyOf) where\n",
            "import qualified Data.Text.Lazy as TL\n",
            "lazyOf :: Int -> TL.Text\n",
            "lazyOf n = TL.pack (show n)\n",
        ),
    )
    .unwrap();
    CompilerBuilder::new()
        .compile_only(true)
        .odir(utf8(odir.path()))
        .hidir(utf8(db.path()))
        .build()
        .unwrap()
        .compile_module_only(utf8(&src.path().join("Producer.hs")))
        .expect("the producer compiles");

    std::fs::write(
        src.path().join("Consumer.hs"),
        concat!(
            "module Consumer (strictOf) where\n",
            "import Data.Text (Text)\n",
            "import Data.Text.Lazy (toStrict)\n",
            "import Producer\n",
            "strictOf :: Int -> Text\n",
            "strictOf n = toStrict (lazyOf n)\n",
        ),
    )
    .unwrap();
    CompilerBuilder::new()
        .compile_only(true)
        .odir(utf8(odir.path()))
        .hidir(utf8(db.path()))
        .package_db(utf8(db.path()))
        .build()
        .unwrap()
        .compile_module_only(utf8(&src.path().join("Consumer.hs")))
        .expect("toStrict of an imported lazy Text checks");
}
