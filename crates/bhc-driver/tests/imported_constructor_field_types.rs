//! Regression: a constructor imported through an interface has its declared
//! field types.
//!
//! Only `Integer` and plain `Text` fields were kept; every other field of a
//! parameterless type was a fresh variable. A module could then put an `Int`
//! in an imported `String` field, and a variable bound by matching the
//! constructor had no type at all (pandoc-types' `walkBlockM` could not tell
//! its `Walkable` dictionaries apart and took the last one).

use bhc_driver::CompilerBuilder;
use camino::Utf8PathBuf;

fn utf8(path: &std::path::Path) -> Utf8PathBuf {
    Utf8PathBuf::from(path.to_str().unwrap())
}

fn compile_consumer(body: &str) -> bool {
    let src = tempfile::tempdir().unwrap();
    let odir = tempfile::tempdir().unwrap();
    let db = tempfile::tempdir().unwrap();

    std::fs::write(
        src.path().join("Shapes.hs"),
        concat!(
            "module Shapes where\n",
            "data Inl = Str String | Emph [Inl]\n",
            "data Blk = Plain [Inl] | Code String | Quote [Blk]\n",
        ),
    )
    .unwrap();
    CompilerBuilder::new()
        .compile_only(true)
        .odir(utf8(odir.path()))
        .hidir(utf8(db.path()))
        .build()
        .unwrap()
        .compile_module_only(utf8(&src.path().join("Shapes.hs")))
        .expect("the producer compiles");

    std::fs::write(src.path().join("Consumer.hs"), body).unwrap();
    CompilerBuilder::new()
        .compile_only(true)
        .odir(utf8(odir.path()))
        .hidir(utf8(db.path()))
        .package_db(utf8(db.path()))
        .build()
        .unwrap()
        .compile_module_only(utf8(&src.path().join("Consumer.hs")))
        .is_ok()
}

#[test]
fn imported_constructor_fields_are_typed() {
    assert!(
        compile_consumer(concat!(
            "module Consumer where\n",
            "import Shapes\n",
            "size :: Blk -> Int\n",
            "size (Plain xs) = length xs\n",
            "size (Code s) = length s\n",
            "size (Quote bs) = sum (map size bs)\n",
            "mk :: Blk\n",
            "mk = Quote [Plain [Str \"a\", Emph []], Code \"c\"]\n",
        )),
        "a well-typed use of the imported constructors compiles"
    );
    assert!(
        !compile_consumer(concat!(
            "module Consumer where\n",
            "import Shapes\n",
            "mk :: Blk\n",
            "mk = Code (3 :: Int)\n",
        )),
        "an Int in an imported String field is a type error"
    );
    assert!(
        !compile_consumer(concat!(
            "module Consumer where\n",
            "import Shapes\n",
            "bad :: Blk -> Int\n",
            "bad (Plain xs) = xs + 1\n",
            "bad _ = 0\n",
        )),
        "a matched field has its declared type, not a fresh one"
    );
}
