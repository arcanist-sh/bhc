//! Regression: class methods and record-field accessors must travel through
//! `.bhi` interfaces to modules compiled against a package DB.
//!
//! `interface_to_module_exports` registered values, types, and constructors
//! but never `iface.classes` or constructor field names — so a module
//! compiled with `--package-db` couldn't resolve an imported class method
//! (Text.Pandoc.Shared failed lowering on Walkable's `walk`/`query`) or a
//! newtype accessor (`B.unMany`), even with the defining module's interface
//! in the DB.

use bhc_driver::CompilerBuilder;
use camino::Utf8PathBuf;

#[test]
fn class_methods_and_accessors_resolve_from_interface() {
    let src_dir = tempfile::tempdir().unwrap();
    let odir = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();

    // Producer: a class with a method, an instance, and a newtype with a
    // record field (the Walkable + Many shape from pandoc-types).
    std::fs::write(
        src_dir.path().join("Walky.hs"),
        concat!(
            "module Walky (Walkable (..), Many (..)) where\n",
            "newtype Many a = Many { unMany :: [a] }\n",
            "class Walkable a where\n",
            "  walk :: (a -> a) -> a -> a\n",
            "instance Walkable Int where\n",
            "  walk f x = f x\n",
        ),
    )
    .unwrap();

    let db_path = Utf8PathBuf::from(db.path().to_str().unwrap());
    let producer = CompilerBuilder::new()
        .compile_only(true)
        .odir(Utf8PathBuf::from(odir.path().to_str().unwrap()))
        .hidir(db_path.clone())
        .build()
        .unwrap();
    producer
        .compile_module_only(Utf8PathBuf::from(
            src_dir.path().join("Walky.hs").to_str().unwrap(),
        ))
        .expect("producer should compile");
    assert!(db.path().join("Walky.bhi").exists());

    // Consumer: uses the class METHOD and the field ACCESSOR; the producer's
    // source is not available — only its interface.
    let consumer_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        consumer_dir.path().join("UseWalky.hs"),
        concat!(
            "module UseWalky (bump, size) where\n",
            "import Walky\n",
            "bump :: Int -> Int\n",
            "bump = walk (+ 1)\n",
            "size :: Many Int -> Int\n",
            "size m = length (unMany m)\n",
        ),
    )
    .unwrap();

    let consumer = CompilerBuilder::new()
        .compile_only(true)
        .odir(Utf8PathBuf::from(odir.path().to_str().unwrap()))
        .hidir(db_path.clone())
        .package_db(db_path)
        .build()
        .unwrap();
    consumer
        .compile_module_only(Utf8PathBuf::from(
            consumer_dir.path().join("UseWalky.hs").to_str().unwrap(),
        ))
        .expect("consumer should resolve walk/unMany from the interface");

    let obj = odir.path().join("UseWalky.o");
    assert!(obj.exists(), "expected object at {}", obj.display());
}
