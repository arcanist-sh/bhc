//! Basel Haskell Compiler (BHC) - Main Entry Point
//!
//! BHC is a next-generation Haskell compiler targeting the Haskell 2026 Platform.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use std::io;
use std::path::{Path, PathBuf};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

/// Basel Haskell Compiler - A next-generation Haskell compiler for 2026
#[derive(Parser, Debug, Clone)]
#[command(name = "bhc")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// The command to execute
    #[command(subcommand)]
    command: Option<Commands>,

    /// Input files to compile
    #[arg(value_name = "FILE")]
    files: Vec<PathBuf>,

    /// Output file name
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Compilation profile
    #[arg(long, value_enum, default_value = "default")]
    profile: Profile,

    /// Haskell edition (accepted for compatibility; not implemented yet)
    #[arg(long)]
    edition: Option<String>,

    /// Optimization level (0-3)
    // Default "2" == OptLevel::Default, matching the session default this flag
    // previously (silently) never overrode — so behavior is unchanged unless
    // the user passes an explicit level.
    #[arg(short = 'O', long, default_value = "2")]
    opt_level: u8,

    /// Increase output verbosity (-v info, -vv debug, -vvv trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Suppress non-error output
    #[arg(short, long)]
    quiet: bool,

    /// Emit kernel fusion report (Numeric profile)
    #[arg(long)]
    kernel_report: bool,

    /// Explicitly enable the tensor fusion pipeline (regardless of profile).
    /// The Numeric profile always fuses; accepted for toolchain compatibility.
    #[arg(long = "tensor-fusion", global = true)]
    tensor_fusion: bool,

    /// Dump an intermediate representation to stdout
    #[arg(long)]
    dump_ir: Option<IrStage>,

    /// Number of parallel jobs (accepted for compatibility; not implemented yet)
    #[arg(short, long)]
    jobs: Option<usize>,

    /// Target triple (e.g., wasm32-wasi, x86_64-unknown-linux-gnu, cuda)
    #[arg(long)]
    target: Option<String>,

    /// Emit format for output: ptx (GPU), or llvm-ir (print the optimized
    /// LLVM IR to stdout, like --dump-ir=llvm)
    #[arg(long)]
    emit: Option<String>,

    /// Additional import paths for module search
    #[arg(short = 'I', long = "import-path", value_name = "DIR")]
    import_paths: Vec<PathBuf>,

    /// Dependency source roots (e.g. a dependency package's `src/`). Their
    /// modules resolve the target's imports during `check` but are not reported.
    /// May be given before or after the subcommand.
    #[arg(long = "package-dir", value_name = "DIR", global = true)]
    package_dirs: Vec<PathBuf>,

    /// Hackage packages to include (format: "name:version")
    #[arg(long = "package", value_name = "PKG:VER")]
    packages: Vec<String>,

    /// Print bare compiler version number (for toolchain detection)
    #[arg(long = "numeric-version", hide = true)]
    numeric_version: bool,

    /// Compile to object file only, do not link
    #[arg(short = 'c', long = "compile-only")]
    compile_only: bool,

    /// Output directory for object files (used with -c)
    #[arg(long = "odir", value_name = "DIR")]
    odir: Option<PathBuf>,

    /// Output directory for interface files (used with -c)
    #[arg(long = "hidir", value_name = "DIR")]
    hidir: Option<PathBuf>,

    /// Package database paths (dirs of compiled `.bhi` interfaces). Imports
    /// resolve from these via separate compilation. May be given before or
    /// after the subcommand.
    #[arg(long = "package-db", value_name = "PATH", global = true)]
    package_dbs: Vec<PathBuf>,

    /// Expose a dependency by package ID, scoping which package-DB packages are
    /// visible. May be given before or after the subcommand.
    #[arg(long = "package-id", value_name = "ID", global = true)]
    package_ids: Vec<String>,

    /// Enable language extensions (e.g., -XOverloadedStrings)
    #[arg(short = 'X', value_name = "EXT", hide = true)]
    extensions: Vec<String>,

    /// CPP preprocessor defines (e.g., -DFOO, -DBAR=1)
    #[arg(short = 'D', value_name = "MACRO[=VALUE]")]
    defines: Vec<String>,

    /// Enable all warnings
    #[arg(long = "Wall", hide = true)]
    wall: bool,

    /// Treat warnings as errors
    #[arg(long = "Werror", hide = true)]
    werror: bool,
}

/// Compilation profile
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Profile {
    /// General Haskell with lazy evaluation
    Default,
    /// Optimized for server workloads
    Server,
    /// Optimized for numeric/tensor computation
    Numeric,
    /// Minimal runtime for embedded/WASM
    Edge,
}

/// IR stages for dumping
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum IrStage {
    /// Abstract syntax tree
    Ast,
    /// High-level IR
    Hir,
    /// Core IR
    Core,
    /// Tensor IR
    Tensor,
    /// Loop IR
    Loop,
    /// LLVM IR (after optimization)
    Llvm,
    /// AST, HIR, Core and LLVM IR
    All,
}

/// Subcommands
#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// Compile source files
    Build {
        /// Input files
        #[arg(value_name = "FILE")]
        files: Vec<PathBuf>,
    },

    /// Check source files without generating code
    Check {
        /// Input files
        #[arg(value_name = "FILE")]
        files: Vec<PathBuf>,
    },

    /// Run a Haskell program
    Run {
        /// The file to run
        file: PathBuf,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Start the interactive REPL
    Repl,

    /// Show version information
    Version,

    /// Generate shell completions
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
    },
}

/// Install a panic hook that reports internal compiler errors with a
/// bug-report banner instead of a raw Rust backtrace.
fn install_ice_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        eprintln!();
        eprintln!("error: internal compiler error: unexpected panic");
        eprintln!();
        eprintln!("note: the compiler unexpectedly panicked. this is a bug.");
        eprintln!();
        eprintln!(
            "note: we would appreciate a bug report: \
             https://github.com/raskell-io/bhc/issues/new"
        );
        eprintln!("note: bhc version: {}", env!("CARGO_PKG_VERSION"));
        eprintln!("note: re-run with RUST_BACKTRACE=1 for a backtrace if one is not shown below");
        eprintln!();
        default_hook(info);
    }));
}

/// Stack for the thread the compiler actually runs on.
///
/// Lowering is recursive over the program's shape, and a module with a very
/// large literal match (pandoc's `Readers.Docx.Symbols`) drives the decision
/// tree deep enough to overflow the 8 MiB a main thread gets by default.
const COMPILER_STACK_SIZE: usize = 1024 * 1024 * 1024;

/// Restore the default SIGPIPE action so `bhc … | head` exits quietly.
///
/// Rust ignores SIGPIPE, which turns a closed pipe into a panic inside
/// `println!`.
#[cfg(unix)]
fn reset_sigpipe() {
    extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }
    const SIGPIPE: i32 = 13;
    const SIG_DFL: usize = 0;
    // SAFETY: installing the default disposition for SIGPIPE before any other
    // thread starts; no handler code runs in this process.
    unsafe {
        signal(SIGPIPE, SIG_DFL);
    }
}

#[cfg(not(unix))]
fn reset_sigpipe() {}

fn main() -> Result<()> {
    reset_sigpipe();
    // Run on a thread with room for deep recursion; the main thread's stack
    // cannot be grown after the process starts.
    let child = std::thread::Builder::new()
        .name("bhc-main".to_string())
        .stack_size(COMPILER_STACK_SIZE)
        .spawn(real_main)
        .expect("failed to spawn compiler thread");
    match child.join() {
        Ok(r) => r,
        // The thread already reported the panic through the ICE hook.
        Err(_) => std::process::exit(101),
    }
}

fn real_main() -> Result<()> {
    install_ice_hook();
    let mut cli = Cli::parse();
    if let Some(target) = cli.target.take() {
        cli.target = Some(normalize_target(&target)?);
    }
    // Validate --dump-ir/--emit up front so a bad value fails before compiling.
    dump_options(&cli)?;
    if cli.edition.is_some() {
        eprintln!("warning: --edition has no effect yet; all modules use bhc's defaults");
    }
    if cli.jobs.is_some() {
        eprintln!("warning: -j/--jobs has no effect yet; modules compile sequentially");
    }

    // --numeric-version: print bare version and exit (used by hx-bhc for toolchain detection)
    if cli.numeric_version {
        println!("{}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    // Set up logging based on verbosity flags
    // Default: WARN (quiet operation), -v: INFO, -vv: DEBUG, -vvv: TRACE
    // --quiet: ERROR only
    let log_level = if cli.quiet {
        Level::ERROR
    } else {
        match cli.verbose {
            0 => Level::WARN,  // Default: only warnings and errors
            1 => Level::INFO,  // -v: informational messages
            2 => Level::DEBUG, // -vv: debug output
            _ => Level::TRACE, // -vvv: trace everything
        }
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Initialize the string interner with common keywords
    bhc_intern::kw::intern_all();

    match cli.command {
        Some(Commands::Build { ref files }) => {
            compile_files(files, &cli)?;
        }
        Some(Commands::Check { ref files }) => {
            check_files(files, &cli)?;
        }
        Some(Commands::Run { ref file, ref args }) => {
            run_file(file, args, &cli)?;
        }
        Some(Commands::Repl) => {
            start_repl(&cli)?;
        }
        Some(Commands::Version) => {
            print_version();
        }
        Some(Commands::Completions { shell }) => {
            generate_completions(shell);
        }
        None => {
            if cli.files.is_empty() {
                // No files specified, print help
                println!(
                    "Basel Haskell Compiler (BHC) v{}",
                    env!("CARGO_PKG_VERSION")
                );
                println!();
                println!("Usage: bhc [OPTIONS] [FILES]...");
                println!();
                println!("For more information, try '--help'");
            } else if cli.compile_only {
                compile_modules_only(&cli.files, &cli)?;
            } else {
                compile_files(&cli.files, &cli)?;
            }
        }
    }

    Ok(())
}

/// Map a `--target` value onto the triple the driver understands.
///
/// Accepts the short names from the documentation (`wasi`, `cuda`, `rocm`,
/// `riscv64`) as well as full triples. An unknown target is an error rather
/// than a silent native build for the host.
fn normalize_target(target: &str) -> Result<String> {
    let normalized = match target {
        "wasi" | "wasm" | "wasm32" | "wasm32-wasi" | "wasm32-wasip1" | "wasm32-unknown-wasi" => {
            "wasm32-wasi"
        }
        "cuda" | "ptx" | "nvptx" | "nvptx64" | "nvptx64-nvidia-cuda" => "nvptx64-nvidia-cuda",
        "rocm" | "amdgcn" | "amdgcn-amd-amdhsa" => "amdgcn-amd-amdhsa",
        "riscv64" => "riscv64-unknown-linux-gnu",
        "riscv32" => "riscv32-unknown-none",
        triple if bhc_target::parse_triple(triple).is_ok() => triple,
        _ => anyhow::bail!(
            "unknown target `{target}`; expected one of wasi, cuda, rocm, riscv64, \
             or a triple such as x86_64-unknown-linux-gnu or aarch64-apple-darwin"
        ),
    };
    Ok(normalized.to_string())
}

/// The IR dumps requested by `--dump-ir` and `--emit llvm-ir`.
fn dump_options(cli: &Cli) -> Result<bhc_session::IrDumpOptions> {
    let mut options = bhc_session::IrDumpOptions::default();
    match cli.dump_ir {
        None => {}
        Some(IrStage::Ast) => options.dump_ast = true,
        Some(IrStage::Hir) => options.dump_hir = true,
        Some(IrStage::Core) => options.dump_core = true,
        Some(IrStage::Llvm) => options.dump_llvm = true,
        Some(IrStage::All) => {
            options.dump_ast = true;
            options.dump_hir = true;
            options.dump_core = true;
            options.dump_llvm = true;
        }
        Some(stage @ (IrStage::Tensor | IrStage::Loop)) => anyhow::bail!(
            "--dump-ir={} is not implemented yet; native code is generated from Core \
             (use --dump-ir=core or --kernel-report)",
            format!("{stage:?}").to_lowercase()
        ),
    }
    match cli.emit.as_deref() {
        None | Some("ptx") => {}
        Some("llvm-ir") => options.dump_llvm = true,
        Some(other) => anyhow::bail!("unknown --emit format `{other}`; expected ptx or llvm-ir"),
    }
    Ok(options)
}

/// Generate shell completions
fn generate_completions(shell: Shell) {
    let mut cmd = Cli::command();
    generate(shell, &mut cmd, "bhc", &mut io::stdout());
}

/// Compile source files
fn compile_files(files: &[PathBuf], cli: &Cli) -> Result<()> {
    try_compile_files(files, cli).map_err(|e| match e.downcast::<CompileFailed>() {
        Ok(CompileFailed(msg)) => {
            eprintln!("error: {}", msg);
            std::process::exit(1);
        }
        Err(other) => other,
    })
}

/// A compile error, already formatted for `error: ...` reporting.
#[derive(Debug)]
struct CompileFailed(String);

impl std::fmt::Display for CompileFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CompileFailed {}

/// [`compile_files`], returning a compile error as [`CompileFailed`] instead
/// of exiting, so a caller can clean up first (`bhc run`).
fn try_compile_files(files: &[PathBuf], cli: &Cli) -> Result<()> {
    use bhc_driver::CompilerBuilder;
    use camino::Utf8PathBuf;

    tracing::info!(
        "Compiling {} file(s) with {:?} profile",
        files.len(),
        cli.profile
    );

    // Convert profile
    let profile = match cli.profile {
        Profile::Default => bhc_session::Profile::Default,
        Profile::Server => bhc_session::Profile::Server,
        Profile::Numeric => bhc_session::Profile::Numeric,
        Profile::Edge => bhc_session::Profile::Edge,
    };

    // Determine output type based on target and emit flags
    let output_type = if cli.emit.as_deref() == Some("ptx") {
        bhc_session::OutputType::Object // PTX is emitted as object
    } else if cli.target.as_deref() == Some("wasm32-wasi") {
        bhc_session::OutputType::Wasm
    } else {
        bhc_session::OutputType::Executable
    };

    // Map the numeric -O flag onto the session optimization level. The flag
    // was previously parsed but never wired, so every compile ran at the
    // session default; honor it now (0=None, 1=Less, 2=Default, 3+=Aggressive).
    let opt_level = match cli.opt_level {
        0 => bhc_session::OptLevel::None,
        1 => bhc_session::OptLevel::Less,
        2 => bhc_session::OptLevel::Default,
        _ => bhc_session::OptLevel::Aggressive,
    };

    // Build compiler with configuration
    let mut builder = CompilerBuilder::new()
        .profile(profile)
        .dump_ir(dump_options(cli)?)
        .output_type(output_type)
        .opt_level(opt_level)
        .emit_kernel_report(cli.kernel_report)
        .tensor_fusion(cli.tensor_fusion)
        // Whole-program builds compile Data.Monoid/Data.Semigroup from the
        // bundled sources; `-c` keeps them builtin (#52).
        .bundled_library(!cli.compile_only);

    // Set target if specified
    if let Some(ref target) = cli.target {
        builder = builder.target(target.clone());
    }

    // Set output path if specified
    if let Some(ref output) = cli.output {
        builder = builder.output_path(
            Utf8PathBuf::from_path_buf(output.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in output path"))?,
        );
    }

    // Add import paths
    for path in &cli.import_paths {
        builder = builder.import_path(
            Utf8PathBuf::from_path_buf(path.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in import path"))?,
        );
    }

    // Add Hackage packages
    for pkg in &cli.packages {
        builder = builder.hackage_package(pkg.clone());
    }

    // Compile-only mode and its output directories (-c, --odir, --hidir), plus
    // package databases for resolving imports via compiled `.bhi` interfaces.
    if cli.compile_only {
        builder = builder.compile_only(true);
    }
    if let Some(ref odir) = cli.odir {
        builder = builder.odir(
            Utf8PathBuf::from_path_buf(odir.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in odir path"))?,
        );
    }
    if let Some(ref hidir) = cli.hidir {
        builder = builder.hidir(
            Utf8PathBuf::from_path_buf(hidir.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in hidir path"))?,
        );
    }
    for db in &cli.package_dbs {
        builder = builder.package_db(
            Utf8PathBuf::from_path_buf(db.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in package-db path"))?,
        );
    }

    // Add CPP defines
    for def in &cli.defines {
        builder = builder.define(def);
    }

    let compiler = builder
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create compiler: {}", e))?;

    // Convert paths and compile
    let utf8_paths: Vec<Utf8PathBuf> = files
        .iter()
        .map(|p| {
            Utf8PathBuf::from_path_buf(p.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in path: {}", p.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    // Use ordered compilation for multiple files, or auto-discovery for single file.
    // In compile-only mode a single file goes through compile_module_only, which
    // honors --odir/--hidir and emits a `.bhi` interface (used to populate an hx
    // package database).
    let compile_result = if utf8_paths.len() > 1 {
        compiler.compile_files_ordered(utf8_paths.iter().map(|p| p.as_path()))
    } else if utf8_paths.len() == 1 {
        if cli.compile_only {
            compiler
                .compile_module_only(utf8_paths[0].as_path())
                .map(|o| vec![o])
        } else {
            compiler.compile_with_discovery(utf8_paths[0].as_path())
        }
    } else {
        compiler.compile_files(utf8_paths.iter().map(|p| p.as_path()))
    };

    match compile_result {
        Ok(outputs) => {
            for output in &outputs {
                tracing::info!("Generated: {}", output.path);
            }
            // Print final output path (always visible unless --quiet)
            if !cli.quiet {
                if let Some(output) = outputs.last() {
                    println!("{}", output.path);
                }
            }
            Ok(())
        }
        Err(e) => Err(CompileFailed(e.to_string()).into()),
    }
}

/// Check source files without generating code
fn check_files(files: &[PathBuf], cli: &Cli) -> Result<()> {
    use bhc_driver::CompilerBuilder;
    use camino::Utf8PathBuf;

    tracing::info!("Checking {} file(s)", files.len());

    let profile = match cli.profile {
        Profile::Default => bhc_session::Profile::Default,
        Profile::Server => bhc_session::Profile::Server,
        Profile::Numeric => bhc_session::Profile::Numeric,
        Profile::Edge => bhc_session::Profile::Edge,
    };

    // Build compiler with import paths and CPP defines
    let mut builder = CompilerBuilder::new()
        .profile(profile)
        .dump_ir(dump_options(cli)?);

    for path in &cli.import_paths {
        builder = builder.import_path(
            Utf8PathBuf::from_path_buf(path.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in import path"))?,
        );
    }

    // Dependency source roots also act as import search paths so on-disk
    // resolution can find their modules in addition to the registry cache.
    let package_dirs: Vec<Utf8PathBuf> = cli
        .package_dirs
        .iter()
        .map(|p| {
            Utf8PathBuf::from_path_buf(p.clone())
                .map_err(|p| anyhow::anyhow!("Invalid UTF-8 in package dir: {}", p.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    for dir in &package_dirs {
        builder = builder.import_path(dir.clone());
    }

    // Package databases (compiled `.bhi` interfaces, e.g. an hx-built DB) and
    // the interface output dir let imports resolve via separate compilation.
    for db in &cli.package_dbs {
        builder = builder.package_db(
            Utf8PathBuf::from_path_buf(db.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in package-db path"))?,
        );
    }
    // Restrict which package-DB packages are visible (by package id). When none
    // are given, every registered package in the DBs is visible.
    for id in &cli.package_ids {
        builder = builder.package_id(id.clone());
    }
    if let Some(ref hidir) = cli.hidir {
        builder = builder.hidir(
            Utf8PathBuf::from_path_buf(hidir.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in hidir path"))?,
        );
    }

    for def in &cli.defines {
        builder = builder.define(def);
    }

    let compiler = builder.build()?;

    // Convert all paths to Utf8PathBuf
    let utf8_paths: Vec<Utf8PathBuf> = files
        .iter()
        .map(|p| {
            Utf8PathBuf::from_path_buf(p.clone())
                .map_err(|p| anyhow::anyhow!("Invalid UTF-8 path: {}", p.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    // Check if any argument is a directory or we have multiple files
    let has_directory = utf8_paths.iter().any(|p| p.as_std_path().is_dir());

    if has_directory || utf8_paths.len() > 1 || !package_dirs.is_empty() {
        // Multi-module check with directory expansion. Dependency roots, when
        // provided, resolve the target's imports without being reported.
        let results = if package_dirs.is_empty() {
            compiler.check_with_discovery(&utf8_paths)?
        } else {
            compiler.check_with_discovery_with_deps(&utf8_paths, &package_dirs)?
        };

        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut skipped = 0usize;

        for (mod_name, result) in &results {
            match result {
                Ok(()) => {
                    println!("  {} OK", mod_name);
                    passed += 1;
                }
                Err(e) => {
                    let msg = e.to_string();
                    if msg.starts_with("skipped:") {
                        eprintln!("  {} SKIPPED: {}", mod_name, msg);
                        skipped += 1;
                    } else {
                        eprintln!("  {} FAILED: {}", mod_name, e);
                        failed += 1;
                    }
                }
            }
        }

        let total = passed + failed + skipped;
        println!(
            "\n{} passed, {} failed, {} skipped (of {} modules)",
            passed, failed, skipped, total
        );

        if failed > 0 {
            anyhow::bail!("Type checking failed");
        }
    } else if utf8_paths.len() == 1 {
        // Single file — use existing check_file
        match compiler.check_file(&utf8_paths[0]) {
            Ok(()) => println!("  {} OK", utf8_paths[0]),
            Err(e) => {
                eprintln!("  {} FAILED: {}", utf8_paths[0], e);
                anyhow::bail!("Type checking failed");
            }
        }
    }

    Ok(())
}

/// Run a Haskell program: compile it natively (the same pipeline as
/// `bhc FILE`) into a temporary directory, then execute it with `args` and
/// exit with its status.
///
/// This used to evaluate the program in the Core interpreter, whose IO and
/// monad dispatch differ from compiled code (`getArgs >>= print` ran in the
/// list monad, #44); `bhc run` now behaves exactly like the compiled program.
fn run_file(file: &Path, args: &[String], cli: &Cli) -> Result<()> {
    tracing::info!("Running {}", file.display());

    let dir = std::env::temp_dir().join(format!("bhc-run-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let exe = dir.join(
        file.file_stem()
            .map(|s| s.to_os_string())
            .unwrap_or_else(|| "main".into()),
    );

    let mut build = cli.clone();
    build.output = Some(exe.clone());
    build.quiet = true;
    build.compile_only = false;
    let compiled = try_compile_files(&[file.to_path_buf()], &build);

    let status = compiled.and_then(|()| {
        std::process::Command::new(&exe)
            .args(args)
            .status()
            .map_err(|e| anyhow::anyhow!("failed to run {}: {}", exe.display(), e))
    });
    let _ = std::fs::remove_dir_all(&dir);
    let status = match status {
        Ok(status) => status,
        Err(e) => match e.downcast::<CompileFailed>() {
            Ok(CompileFailed(msg)) => {
                eprintln!("error: {}", msg);
                std::process::exit(1);
            }
            Err(other) => return Err(other),
        },
    };
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

/// Compile source files to object files only (no linking), with optional interface generation.
fn compile_modules_only(files: &[PathBuf], cli: &Cli) -> Result<()> {
    use bhc_driver::CompilerBuilder;
    use camino::Utf8PathBuf;

    tracing::info!(
        "Compiling {} file(s) to object files (compile-only mode)",
        files.len()
    );

    let profile = match cli.profile {
        Profile::Default => bhc_session::Profile::Default,
        Profile::Server => bhc_session::Profile::Server,
        Profile::Numeric => bhc_session::Profile::Numeric,
        Profile::Edge => bhc_session::Profile::Edge,
    };

    let mut builder = CompilerBuilder::new()
        .profile(profile)
        .dump_ir(dump_options(cli)?)
        .compile_only(true)
        .output_type(bhc_session::OutputType::Object)
        .emit_kernel_report(cli.kernel_report)
        .tensor_fusion(cli.tensor_fusion);

    if let Some(ref target) = cli.target {
        builder = builder.target(target.clone());
    }

    if let Some(ref odir) = cli.odir {
        builder = builder.odir(
            Utf8PathBuf::from_path_buf(odir.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in odir path"))?,
        );
    }

    if let Some(ref hidir) = cli.hidir {
        builder = builder.hidir(
            Utf8PathBuf::from_path_buf(hidir.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in hidir path"))?,
        );
    }

    for path in &cli.import_paths {
        builder = builder.import_path(
            Utf8PathBuf::from_path_buf(path.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in import path"))?,
        );
    }

    for db in &cli.package_dbs {
        builder = builder.package_db(
            Utf8PathBuf::from_path_buf(db.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in package-db path"))?,
        );
    }

    for id in &cli.package_ids {
        builder = builder.package_id(id.clone());
    }

    // Add CPP defines
    for def in &cli.defines {
        builder = builder.define(def);
    }

    let compiler = builder
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create compiler: {}", e))?;

    let utf8_paths: Vec<Utf8PathBuf> = files
        .iter()
        .map(|p| {
            Utf8PathBuf::from_path_buf(p.clone())
                .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in path: {}", p.display()))
        })
        .collect::<Result<Vec<_>>>()?;

    // Use compile_files_ordered for multi-module context sharing
    // (cross-module record updates, constructor imports, etc.)
    match compiler.compile_files_ordered(utf8_paths.iter().map(|p| p.as_path())) {
        Ok(outputs) => {
            for output in &outputs {
                tracing::info!("Generated: {}", output.path);
                if !cli.quiet {
                    println!("{}", output.path);
                }
            }
        }
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Start the interactive REPL
fn start_repl(cli: &Cli) -> Result<()> {
    use std::process::Command;

    let bhc_exe = std::env::current_exe()?;
    let bhci_exe = bhc_exe.with_file_name("bhci");
    if !bhci_exe.exists() {
        // Release archives ship only `bhc` so far (#50).
        anyhow::bail!(
            "the REPL needs `bhci` next to `bhc` ({} is missing); release builds do not \
             include it yet, so build it from source with `cargo build --release -p bhci` \
             and copy target/release/bhci next to bhc",
            bhci_exe.display()
        );
    }

    let mut args: Vec<String> = Vec::new();
    match cli.profile {
        Profile::Numeric => args.extend(["--profile".into(), "numeric".into()]),
        Profile::Server => args.extend(["--profile".into(), "server".into()]),
        Profile::Edge => args.extend(["--profile".into(), "edge".into()]),
        Profile::Default => {}
    }
    // Forward package databases, exposed package ids, and import paths so the
    // REPL resolves imports against the same separate-compilation interfaces as
    // `bhc check`/`build`.
    for path in &cli.import_paths {
        args.push("--import-path".into());
        args.push(path.display().to_string());
    }
    for db in &cli.package_dbs {
        args.push("--package-db".into());
        args.push(db.display().to_string());
    }
    for id in &cli.package_ids {
        args.push("--package-id".into());
        args.push(id.clone());
    }

    let status = Command::new(&bhci_exe)
        .args(&args)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to start bhci at {}: {}", bhci_exe.display(), e))?;

    std::process::exit(status.code().unwrap_or(1));
}

/// Print version information
fn print_version() {
    println!("Basel Haskell Compiler (BHC)");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!();
    println!("Targets Haskell 2026 Platform Specification");
    println!("Supported profiles: Default, Server, Numeric, Edge");
    println!();
    println!("Repository: https://github.com/raskell-io/bhc");
}

#[cfg(test)]
mod tests {
    use super::normalize_target;

    #[test]
    fn target_aliases_normalize() {
        assert_eq!(normalize_target("wasi").unwrap(), "wasm32-wasi");
        assert_eq!(normalize_target("wasm32-wasip1").unwrap(), "wasm32-wasi");
        assert_eq!(normalize_target("cuda").unwrap(), "nvptx64-nvidia-cuda");
        assert_eq!(normalize_target("rocm").unwrap(), "amdgcn-amd-amdhsa");
        assert_eq!(
            normalize_target("x86_64-unknown-linux-gnu").unwrap(),
            "x86_64-unknown-linux-gnu"
        );
    }

    #[test]
    fn dump_ir_and_emit_select_dumps() {
        use clap::Parser;
        let cli = super::Cli::parse_from(["bhc", "--dump-ir", "core", "M.hs"]);
        let opts = super::dump_options(&cli).unwrap();
        assert!(opts.dump_core && !opts.dump_llvm);
        let cli = super::Cli::parse_from(["bhc", "--emit", "llvm-ir", "M.hs"]);
        assert!(super::dump_options(&cli).unwrap().dump_llvm);
        let cli = super::Cli::parse_from(["bhc", "--dump-ir", "tensor", "M.hs"]);
        assert!(super::dump_options(&cli).is_err());
        let cli = super::Cli::parse_from(["bhc", "--emit", "wat", "M.hs"]);
        assert!(super::dump_options(&cli).is_err());
    }

    #[test]
    fn unknown_target_is_rejected() {
        assert!(normalize_target("bogus").is_err());
        assert!(normalize_target("sparc-sun-solaris").is_err());
    }
}
