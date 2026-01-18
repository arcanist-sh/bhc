//! Basel Haskell Compiler (BHC) - Main Entry Point
//!
//! BHC is a next-generation Haskell compiler targeting the Haskell 2026 Platform.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

/// Basel Haskell Compiler - A next-generation Haskell compiler for 2026
#[derive(Parser, Debug)]
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

    /// Haskell edition
    #[arg(long, default_value = "2026")]
    edition: String,

    /// Optimization level (0-3)
    #[arg(short = 'O', long, default_value = "0")]
    opt_level: u8,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Emit kernel fusion report (Numeric profile)
    #[arg(long)]
    kernel_report: bool,

    /// Dump intermediate representations
    #[arg(long)]
    dump_ir: Option<IrStage>,

    /// Number of parallel jobs
    #[arg(short, long)]
    jobs: Option<usize>,
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
    /// All stages
    All,
}

/// Subcommands
#[derive(Subcommand, Debug)]
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let log_level = if cli.verbose { Level::DEBUG } else { Level::INFO };
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
        None => {
            if cli.files.is_empty() {
                // No files specified, print help
                println!("Basel Haskell Compiler (BHC) v{}", env!("CARGO_PKG_VERSION"));
                println!();
                println!("Usage: bhc [OPTIONS] [FILES]...");
                println!();
                println!("For more information, try '--help'");
            } else {
                compile_files(&cli.files, &cli)?;
            }
        }
    }

    Ok(())
}

/// Compile source files
fn compile_files(files: &[PathBuf], cli: &Cli) -> Result<()> {
    use bhc_diagnostics::{DiagnosticRenderer, SourceMap};
    use bhc_parser::parse_module;
    use bhc_span::FileId;

    tracing::info!("Compiling {} file(s) with {:?} profile", files.len(), cli.profile);

    let mut source_map = SourceMap::new();
    let mut has_errors = false;

    for (i, file) in files.iter().enumerate() {
        let src = std::fs::read_to_string(file)?;
        let file_id = source_map.add_file(file.display().to_string(), src.clone());

        tracing::debug!("Parsing {}", file.display());

        let (module, diagnostics) = parse_module(&src, file_id);

        // Render diagnostics
        let renderer = DiagnosticRenderer::new(&source_map);
        renderer.render_all(&diagnostics);

        if diagnostics.iter().any(|d| d.is_error()) {
            has_errors = true;
            continue;
        }

        if let Some(module) = module {
            tracing::info!(
                "Parsed module: {}",
                module.name.as_ref().map_or("<main>".to_string(), |n| n.to_string())
            );

            // Dump AST if requested
            if matches!(cli.dump_ir, Some(IrStage::Ast | IrStage::All)) {
                println!("=== AST for {} ===", file.display());
                println!("{:#?}", module);
            }

            // TODO: Continue compilation pipeline
            // 1. Desugar to HIR
            // 2. Type check
            // 3. Lower to Core IR
            // 4. (Numeric profile) Lower to Tensor IR
            // 5. (Numeric profile) Lower to Loop IR
            // 6. Code generation
        }
    }

    if has_errors {
        std::process::exit(1);
    }

    Ok(())
}

/// Check source files without generating code
fn check_files(files: &[PathBuf], cli: &Cli) -> Result<()> {
    tracing::info!("Checking {} file(s)", files.len());
    // For now, just parse
    compile_files(files, cli)
}

/// Run a Haskell program
fn run_file(file: &PathBuf, _args: &[String], cli: &Cli) -> Result<()> {
    use bhc_driver::CompilerBuilder;
    use camino::Utf8PathBuf;

    tracing::info!("Running {}", file.display());

    // Convert profile
    let profile = match cli.profile {
        Profile::Default => bhc_session::Profile::Default,
        Profile::Server => bhc_session::Profile::Server,
        Profile::Numeric => bhc_session::Profile::Numeric,
        Profile::Edge => bhc_session::Profile::Edge,
    };

    // Build compiler
    let compiler = CompilerBuilder::new()
        .profile(profile)
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create compiler: {}", e))?;

    // Convert path
    let path = Utf8PathBuf::from_path_buf(file.clone())
        .map_err(|_| anyhow::anyhow!("Invalid UTF-8 in path"))?;

    // Run the file
    match compiler.run_file(&path) {
        Ok(value) => {
            println!("{:?}", value);
            Ok(())
        }
        Err(e) => {
            eprintln!("Execution error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Start the interactive REPL
fn start_repl(_cli: &Cli) -> Result<()> {
    println!("Basel Haskell Compiler Interactive (bhci)");
    println!("Type :help for help, :quit to exit");
    println!();
    // TODO: Implement REPL
    println!("(REPL not yet implemented)");
    Ok(())
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
