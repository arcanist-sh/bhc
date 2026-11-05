//! Using a concurrency primitive that would abort at run time warns at compile
//! time (#38).
//!
//! Lowering reports each one used, once per name, as
//! `LowerWarning::NotImplemented`. It stays a warning: a module that merely
//! mentions them (pandoc's `Text.Pandoc.Process`) must still compile. Threads,
//! `MVar`s and `Chan`s run under the closure IO model, so they warn only under
//! the eager model; `throwTo` warns under both. A user's own function of the
//! same name is not flagged.

use bhc_lower::{LowerConfig, LowerContext, LowerWarning};
use bhc_parser::parse_module;
use bhc_span::FileId;

/// The names `NotImplemented` warnings were reported for, in order.
fn not_implemented_warnings(source: &str) -> Vec<String> {
    let (module, diags) = parse_module(source, FileId::new(0));
    assert!(
        diags
            .iter()
            .all(|d| d.severity != bhc_diagnostics::Severity::Error),
        "parse errors: {diags:?}"
    );
    let module = module.expect("module");
    let config = LowerConfig {
        include_builtins: true,
        warn_unused: false,
        search_paths: vec![],
    };
    let mut ctx = LowerContext::with_builtins();
    bhc_lower::lower_module(&mut ctx, &module, &config).expect("lowering succeeds");
    ctx.warnings
        .iter()
        .filter_map(|w| match w {
            LowerWarning::NotImplemented { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// The warnings expected for `names`: only `throwTo` under the closure model.
fn expected<'a>(names: &[&'a str]) -> Vec<&'a str> {
    let eager = std::env::var("BHC_IO_MODEL").as_deref() == Ok("eager");
    names
        .iter()
        .copied()
        .filter(|name| eager || *name == "throwTo")
        .collect()
}

#[test]
fn concurrency_primitives_warn_once_each() {
    let names = not_implemented_warnings(concat!(
        "import Control.Concurrent\n",
        "main = do\n",
        "  mv <- newEmptyMVar\n",
        "  _ <- forkIO (putMVar mv (42 :: Int))\n",
        "  t <- forkIO (putMVar mv 43)\n",
        "  throwTo t (userError \"x\")\n",
        "  throwTo t (userError \"y\")\n",
        "  takeMVar mv >>= print\n",
    ));
    assert_eq!(
        names,
        expected(&["newEmptyMVar", "forkIO", "putMVar", "throwTo", "takeMVar"])
    );
}

#[test]
fn qualified_concurrency_primitives_warn() {
    let names = not_implemented_warnings(concat!(
        "import qualified Control.Concurrent.MVar as MV\n",
        "main = MV.newEmptyMVar >>= \\v -> MV.putMVar v (1 :: Int)\n",
    ));
    assert_eq!(names, expected(&["newEmptyMVar", "putMVar"]));
}

#[test]
fn user_function_of_the_same_name_does_not_warn() {
    let names = not_implemented_warnings(concat!(
        "forkIO :: Int -> Int\n",
        "forkIO x = x + 1\n",
        "main = print (forkIO 1)\n",
    ));
    assert!(names.is_empty(), "unexpected warnings: {names:?}");
}
