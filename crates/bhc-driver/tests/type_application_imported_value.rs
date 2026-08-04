//! Regression: an explicit type application on an IMPORTED value must not pin
//! the whole expression to the type argument.
//!
//! Imported values without an interface type get the permissive placeholder
//! scheme `forall a. a`. `Expr::TypeApp` inference substituted the explicit
//! type arg into the scheme's first forall var — for the placeholder that IS
//! the whole type, so `tshow @Double (0.9 / fromIntegral n)` typed `tshow` as
//! `Double` and failed to apply it (`expected Double, found (t -> t)` in
//! Text.Pandoc.Writers.LaTeX's `toSubfigure`). The fix ignores the type
//! application when the base's scheme is that degenerate placeholder.

use bhc_driver::Compiler;
use camino::Utf8PathBuf;

#[test]
fn type_application_on_imported_value_still_applies() {
    let dep_dir = tempfile::tempdir().unwrap();
    let app_dir = tempfile::tempdir().unwrap();

    std::fs::write(
        dep_dir.path().join("Shr.hs"),
        "module Shr (tshow) where\n\
         import qualified Data.Text as T\n\
         tshow :: Show a => a -> T.Text\n\
         tshow = T.pack . show\n",
    )
    .unwrap();

    std::fs::write(
        app_dir.path().join("Main.hs"),
        "module Main (main) where\n\
         import qualified Data.Text as T\n\
         import Shr (tshow)\n\
         f :: Int -> T.Text\n\
         f n = tshow @Double (0.9 / fromIntegral n)\n\
         main :: IO ()\n\
         main = putStrLn (T.unpack (f 3))\n",
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
        "`tshow @Double x` must apply, not pin `tshow` itself to Double: {:?}",
        main.1
    );
}
