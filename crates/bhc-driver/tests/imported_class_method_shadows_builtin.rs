//! Regression: an explicitly imported class method must shadow a same-named
//! hardcoded builtin.
//!
//! `Text.Pandoc.Class` defines `lookupEnv` as a `PandocMonad` class method
//! (`PandocMonad m => Text -> m (Maybe Text)`), imported into
//! `Text.Pandoc.Readers.LaTeX` via `PandocMonad(..)`. bhc also registers a
//! `System.Environment.lookupEnv` builtin (`String -> IO (Maybe String)`) that
//! is always in scope. The import binding was skipped whenever a name was
//! already bound (`ctx.lookup_value(name).is_none()`), so the builtin won and
//! its `IO` scheme leaked into the class method's use sites. That broke
//! `readFileFromTexinputs`:
//!   `... <$> lookupEnv "TEXINPUTS"` expected `LP m (Maybe Text)` (a `ParsecT`)
//!   but the builtin made it `IO (Maybe String)`.
//!
//! The class-param usage (`m [String]`) happened to slip through by unifying the
//! rigid `m` with `IO`; the failure only surfaced at an *instance* type
//! (`ParsecT ... m`), which cannot unify with `IO`.

use bhc_driver::Compiler;
use camino::Utf8PathBuf;
use std::io::Write;

#[test]
fn imported_class_method_named_like_builtin_wins_at_instance_type() {
    let dep_dir = tempfile::tempdir().expect("dep tempdir");
    let app_dir = tempfile::tempdir().expect("app tempdir");

    // Dependency package: a class whose method is named like a builtin.
    let my_dir = dep_dir.path().join("My");
    std::fs::create_dir_all(&my_dir).expect("mkdir");
    std::fs::write(
        my_dir.join("Cls.hs"),
        "module My.Cls (MyMonad(..)) where\n\
         class Monad m => MyMonad m where\n  \
             lookupEnv :: String -> m (Maybe String)\n",
    )
    .expect("write dep");

    // User module: uses the imported method at an *instance* type.
    std::fs::write(
        app_dir.path().join("User.hs"),
        "{-# LANGUAGE FlexibleContexts #-}\n\
         module User where\n\
         import My.Cls (MyMonad(..))\n\
         newtype WrapT m a = WrapT (m a)\n\
         instance MyMonad m => MyMonad (WrapT m) where\n  \
             lookupEnv s = WrapT (lookupEnv s)\n\
         bar :: MyMonad m => WrapT m [String]\n\
         bar = fmap (maybe [] (:[])) (lookupEnv \"X\")\n",
    )
    .expect("write user");

    let compiler = Compiler::with_defaults().expect("compiler");
    let app = Utf8PathBuf::from(app_dir.path().to_str().unwrap());
    let dep = Utf8PathBuf::from(dep_dir.path().to_str().unwrap());

    let resolved = compiler
        .check_with_discovery_with_deps(&[app], &[dep])
        .expect("discovery");
    let user = resolved
        .iter()
        .find(|(n, _)| n == "User")
        .expect("User module reported");
    assert!(
        user.1.is_ok(),
        "imported class method should shadow the builtin: {:?}",
        user.1
    );
}

/// The builtin must still be usable when it is NOT shadowed by an import.
#[test]
fn system_environment_lookup_env_builtin_still_works() {
    let mut file = tempfile::Builder::new()
        .suffix(".hs")
        .tempfile()
        .expect("temp");
    file.write_all(
        b"module EnvReal where\n\
          import System.Environment (lookupEnv)\n\
          check :: IO (Maybe String)\n\
          check = lookupEnv \"X\"\n",
    )
    .expect("write");
    let path = camino::Utf8Path::from_path(file.path()).expect("utf8");

    let compiler = Compiler::with_defaults().expect("compiler");
    assert!(
        compiler.check_file(path).is_ok(),
        "System.Environment.lookupEnv should still type-check as String -> IO (Maybe String)"
    );
}
