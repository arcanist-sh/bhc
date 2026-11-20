//! Using a concurrency primitive that would abort at run time warns at compile
//! time (#38): on WebAssembly, which has no threads yet (#75). Native targets
//! run them under either IO model.
//!
//! Lowering reports each one used, once per name, as
//! `LowerWarning::NotImplemented`. It stays a warning: a module that merely
//! mentions them (pandoc's `Text.Pandoc.Process`) must still compile. A user's
//! own function of the same name is not flagged.
//!
//! Every test here compiles for a WebAssembly target. The target is fixed
//! once per process, which is why this is its own test binary.

use bhc_lower::{LowerConfig, LowerContext, LowerWarning};
use bhc_parser::parse_module;
use bhc_span::FileId;

/// The names `NotImplemented` warnings were reported for, in order.
fn not_implemented_warnings(source: &str) -> Vec<String> {
    bhc_types::select_io_model_for_target(Some("wasm32-wasi"));
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
        ["newEmptyMVar", "forkIO", "putMVar", "throwTo", "takeMVar"]
    );
}

#[test]
fn qualified_concurrency_primitives_warn() {
    let names = not_implemented_warnings(concat!(
        "import qualified Control.Concurrent.MVar as MV\n",
        "main = MV.newEmptyMVar >>= \\v -> MV.putMVar v (1 :: Int)\n",
    ));
    assert_eq!(names, ["newEmptyMVar", "putMVar"]);
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
