//! Linking support for BHC.
//!
//! This crate handles the final linking stage of compilation, combining
//! object files and libraries into executables or shared libraries.
//!
//! # Overview
//!
//! The linker module supports:
//!
//! - Invoking the system linker (ld, lld, link.exe)
//! - Managing library search paths
//! - Handling platform-specific linking requirements
//! - Static and dynamic linking
//!
//! # Platform Support
//!
//! | Platform | Default Linker | Notes |
//! |----------|---------------|-------|
//! | Linux    | `cc` (gcc/clang) | Uses system linker driver |
//! | macOS    | `cc` (clang) | Uses ld64 underneath |
//! | Windows  | `link.exe` | MSVC linker |
//! | WASM     | `wasm-ld` | LLVM WASM linker |

#![warn(missing_docs)]

use bhc_session::OutputType;
use bhc_target::{Os, TargetSpec};
use camino::{Utf8Path, Utf8PathBuf};
use std::process::Command;
use thiserror::Error;
use tracing::{debug, info, instrument};

/// Errors that can occur during linking.
#[derive(Debug, Error)]
pub enum LinkerError {
    /// The linker executable was not found.
    #[error("linker not found: {0}")]
    LinkerNotFound(String),

    /// The linker failed with an error.
    #[error("linker failed: {message}")]
    LinkerFailed {
        /// Error message from the linker.
        message: String,
        /// Exit code if available.
        exit_code: Option<i32>,
    },

    /// Failed to execute the linker.
    #[error("failed to execute linker: {0}")]
    ExecutionFailed(#[from] std::io::Error),

    /// Invalid linker configuration.
    #[error("invalid linker configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for linker operations.
pub type LinkerResult<T> = Result<T, LinkerError>;

/// Type of output to produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkOutputType {
    /// Executable binary.
    Executable,
    /// Static library.
    StaticLib,
    /// Dynamic/shared library.
    DynamicLib,
}

impl From<OutputType> for LinkOutputType {
    fn from(value: OutputType) -> Self {
        match value {
            OutputType::Executable => Self::Executable,
            OutputType::StaticLib => Self::StaticLib,
            OutputType::DynamicLib => Self::DynamicLib,
            _ => Self::Executable,
        }
    }
}

/// A library to link against.
#[derive(Clone, Debug)]
pub struct LinkLibrary {
    /// Library name (without lib prefix or extension).
    pub name: String,
    /// Whether to link statically.
    pub static_link: bool,
    /// Optional specific path to the library.
    pub path: Option<Utf8PathBuf>,
}

impl LinkLibrary {
    /// Create a new library reference by name.
    #[must_use]
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            static_link: false,
            path: None,
        }
    }

    /// Create a static library reference.
    #[must_use]
    pub fn static_lib(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            static_link: true,
            path: None,
        }
    }

    /// Create a library reference with a specific path.
    #[must_use]
    pub fn with_path(name: impl Into<String>, path: impl Into<Utf8PathBuf>) -> Self {
        Self {
            name: name.into(),
            static_link: false,
            path: Some(path.into()),
        }
    }
}

/// Configuration for the linker.
#[derive(Clone, Debug)]
pub struct LinkerConfig {
    /// Target specification.
    pub target: TargetSpec,
    /// Output type.
    pub output_type: LinkOutputType,
    /// Output path.
    pub output_path: Utf8PathBuf,
    /// Object files to link.
    pub objects: Vec<Utf8PathBuf>,
    /// Libraries to link.
    pub libraries: Vec<LinkLibrary>,
    /// Library search paths.
    pub library_paths: Vec<Utf8PathBuf>,
    /// Generate position-independent executable.
    pub pie: bool,
    /// Enable link-time optimization.
    pub lto: bool,
    /// Strip debug symbols.
    pub strip: bool,
    /// Additional linker flags.
    pub extra_flags: Vec<String>,
}

impl LinkerConfig {
    /// Create a new linker configuration for the given target.
    #[must_use]
    pub fn new(target: TargetSpec, output_path: impl Into<Utf8PathBuf>) -> Self {
        Self {
            target,
            output_type: LinkOutputType::Executable,
            output_path: output_path.into(),
            objects: Vec::new(),
            libraries: Vec::new(),
            library_paths: Vec::new(),
            pie: false,
            lto: false,
            strip: false,
            extra_flags: Vec::new(),
        }
    }

    /// Add an object file to link.
    #[must_use]
    pub fn with_object(mut self, path: impl Into<Utf8PathBuf>) -> Self {
        self.objects.push(path.into());
        self
    }

    /// Add multiple object files.
    #[must_use]
    pub fn with_objects(mut self, paths: impl IntoIterator<Item = impl Into<Utf8PathBuf>>) -> Self {
        self.objects.extend(paths.into_iter().map(Into::into));
        self
    }

    /// Add a library to link.
    #[must_use]
    pub fn with_library(mut self, lib: LinkLibrary) -> Self {
        self.libraries.push(lib);
        self
    }

    /// Add a library search path.
    #[must_use]
    pub fn with_library_path(mut self, path: impl Into<Utf8PathBuf>) -> Self {
        self.library_paths.push(path.into());
        self
    }

    /// Set the output type.
    #[must_use]
    pub fn output_type(mut self, output_type: LinkOutputType) -> Self {
        self.output_type = output_type;
        self
    }

    /// Enable PIE.
    #[must_use]
    pub fn with_pie(mut self, pie: bool) -> Self {
        self.pie = pie;
        self
    }

    /// Enable LTO.
    #[must_use]
    pub fn with_lto(mut self, lto: bool) -> Self {
        self.lto = lto;
        self
    }

    /// Enable symbol stripping.
    #[must_use]
    pub fn with_strip(mut self, strip: bool) -> Self {
        self.strip = strip;
        self
    }
}

/// Trait for linker implementations.
pub trait Linker: Send + Sync {
    /// Get the linker name.
    fn name(&self) -> &'static str;

    /// Check if this linker supports the given target.
    fn supports_target(&self, target: &TargetSpec) -> bool;

    /// Get the linker executable path.
    fn executable(&self) -> &str;

    /// Build the command-line arguments for linking.
    fn build_args(&self, config: &LinkerConfig) -> Vec<String>;

    /// Run the linker with the given configuration.
    fn link(&self, config: &LinkerConfig) -> LinkerResult<()>;
}

/// System linker (cc/gcc/clang on Unix, link.exe on Windows).
#[derive(Clone, Debug)]
pub struct SystemLinker {
    /// Path to the linker executable.
    executable: String,
}

impl SystemLinker {
    /// Create a new system linker for the given target.
    #[must_use]
    pub fn for_target(target: &TargetSpec) -> Self {
        let executable = match target.os {
            Os::Windows => "link.exe".to_string(),
            Os::Wasi => "wasm-ld".to_string(),
            _ => "cc".to_string(),
        };
        Self { executable }
    }

    /// Create a system linker with a specific executable.
    #[must_use]
    pub fn with_executable(executable: impl Into<String>) -> Self {
        Self {
            executable: executable.into(),
        }
    }
}

impl Linker for SystemLinker {
    fn name(&self) -> &'static str {
        "system"
    }

    fn supports_target(&self, _target: &TargetSpec) -> bool {
        true
    }

    fn executable(&self) -> &str {
        &self.executable
    }

    fn build_args(&self, config: &LinkerConfig) -> Vec<String> {
        let mut args = Vec::new();

        match config.target.os {
            Os::Windows => {
                self.build_windows_args(&mut args, config);
            }
            Os::Wasi => {
                self.build_wasm_args(&mut args, config);
            }
            _ => {
                self.build_unix_args(&mut args, config);
            }
        }

        args
    }

    #[instrument(skip(self, config), fields(output = %config.output_path))]
    fn link(&self, config: &LinkerConfig) -> LinkerResult<()> {
        let args = self.build_args(config);

        debug!(linker = %self.executable, args = ?args, "invoking linker");

        let output = Command::new(&self.executable)
            .args(&args)
            .output()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    LinkerError::LinkerNotFound(self.executable.clone())
                } else {
                    LinkerError::ExecutionFailed(e)
                }
            })?;

        if output.status.success() {
            info!(output = %config.output_path, "linking successful");
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(LinkerError::LinkerFailed {
                message: stderr.to_string(),
                exit_code: output.status.code(),
            })
        }
    }
}

impl SystemLinker {
    fn build_unix_args(&self, args: &mut Vec<String>, config: &LinkerConfig) {
        // Output file
        args.push("-o".to_string());
        args.push(config.output_path.to_string());

        // Output type
        match config.output_type {
            LinkOutputType::DynamicLib => {
                args.push("-shared".to_string());
            }
            LinkOutputType::StaticLib => {
                // Static libs use ar, not the linker
                // This would be handled separately
            }
            LinkOutputType::Executable => {
                if config.pie {
                    args.push("-pie".to_string());
                }
            }
        }

        // Object files
        for obj in &config.objects {
            args.push(obj.to_string());
        }

        // Library paths
        for path in &config.library_paths {
            args.push(format!("-L{path}"));
        }

        // Libraries
        for lib in &config.libraries {
            if lib.static_link {
                args.push("-Wl,-Bstatic".to_string());
            }
            if let Some(ref path) = lib.path {
                args.push(path.to_string());
            } else {
                args.push(format!("-l{}", lib.name));
            }
            if lib.static_link {
                args.push("-Wl,-Bdynamic".to_string());
            }
        }

        // LTO
        if config.lto {
            args.push("-flto".to_string());
        }

        // Strip
        if config.strip {
            args.push("-s".to_string());
        }

        // Extra flags
        args.extend(config.extra_flags.iter().cloned());
    }

    fn build_windows_args(&self, args: &mut Vec<String>, config: &LinkerConfig) {
        // Output file
        args.push(format!("/OUT:{}", config.output_path));

        // Output type
        match config.output_type {
            LinkOutputType::DynamicLib => {
                args.push("/DLL".to_string());
            }
            _ => {}
        }

        // Object files
        for obj in &config.objects {
            args.push(obj.to_string());
        }

        // Library paths
        for path in &config.library_paths {
            args.push(format!("/LIBPATH:{path}"));
        }

        // Libraries
        for lib in &config.libraries {
            if let Some(ref path) = lib.path {
                args.push(path.to_string());
            } else {
                args.push(format!("{}.lib", lib.name));
            }
        }

        // Extra flags
        args.extend(config.extra_flags.iter().cloned());
    }

    fn build_wasm_args(&self, args: &mut Vec<String>, config: &LinkerConfig) {
        // Output file
        args.push("-o".to_string());
        args.push(config.output_path.to_string());

        // Object files
        for obj in &config.objects {
            args.push(obj.to_string());
        }

        // WASM-specific flags
        args.push("--no-entry".to_string());
        args.push("--export-all".to_string());

        // Extra flags
        args.extend(config.extra_flags.iter().cloned());
    }
}

/// LLD linker (LLVM's linker).
#[derive(Clone, Debug)]
pub struct LldLinker {
    /// Flavor of LLD to use.
    flavor: LldFlavor,
}

/// LLD flavor (determines command-line interface).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LldFlavor {
    /// GNU-style ld.lld.
    Gnu,
    /// Darwin-style ld64.lld.
    Darwin,
    /// Windows-style lld-link.
    Msvc,
    /// WebAssembly wasm-ld.
    Wasm,
}

impl LldLinker {
    /// Create an LLD linker for the given target.
    #[must_use]
    pub fn for_target(target: &TargetSpec) -> Self {
        let flavor = match target.os {
            Os::MacOs => LldFlavor::Darwin,
            Os::Windows => LldFlavor::Msvc,
            Os::Wasi => LldFlavor::Wasm,
            _ => LldFlavor::Gnu,
        };
        Self { flavor }
    }

    fn executable_name(&self) -> &'static str {
        match self.flavor {
            LldFlavor::Gnu => "ld.lld",
            LldFlavor::Darwin => "ld64.lld",
            LldFlavor::Msvc => "lld-link",
            LldFlavor::Wasm => "wasm-ld",
        }
    }
}

impl Linker for LldLinker {
    fn name(&self) -> &'static str {
        "lld"
    }

    fn supports_target(&self, _target: &TargetSpec) -> bool {
        true
    }

    fn executable(&self) -> &str {
        self.executable_name()
    }

    fn build_args(&self, config: &LinkerConfig) -> Vec<String> {
        // LLD uses similar arguments to the system linker for each flavor
        let system = SystemLinker::with_executable(self.executable_name());
        system.build_args(config)
    }

    fn link(&self, config: &LinkerConfig) -> LinkerResult<()> {
        let system = SystemLinker::with_executable(self.executable_name());
        system.link(config)
    }
}

/// Create the appropriate linker for a target.
#[must_use]
pub fn linker_for_target(target: &TargetSpec) -> Box<dyn Linker> {
    Box::new(SystemLinker::for_target(target))
}

/// Perform linking with the given configuration.
///
/// # Errors
///
/// Returns an error if linking fails.
#[instrument(skip(config), fields(output = %config.output_path))]
pub fn link(config: &LinkerConfig) -> LinkerResult<()> {
    let linker = linker_for_target(&config.target);
    linker.link(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_target::targets;

    #[test]
    fn test_linker_config() {
        let target = targets::x86_64_linux_gnu();
        let config = LinkerConfig::new(target, "output")
            .with_object("main.o")
            .with_library(LinkLibrary::named("m"))
            .with_pie(true);

        assert_eq!(config.objects.len(), 1);
        assert_eq!(config.libraries.len(), 1);
        assert!(config.pie);
    }

    #[test]
    fn test_unix_args() {
        let target = targets::x86_64_linux_gnu();
        let config = LinkerConfig::new(target.clone(), "a.out")
            .with_object("main.o")
            .with_library(LinkLibrary::named("c"));

        let linker = SystemLinker::for_target(&target);
        let args = linker.build_args(&config);

        assert!(args.contains(&"-o".to_string()));
        assert!(args.contains(&"a.out".to_string()));
        assert!(args.contains(&"main.o".to_string()));
        assert!(args.contains(&"-lc".to_string()));
    }

    #[test]
    fn test_link_library() {
        let lib = LinkLibrary::named("pthread");
        assert_eq!(lib.name, "pthread");
        assert!(!lib.static_link);

        let static_lib = LinkLibrary::static_lib("mylib");
        assert!(static_lib.static_link);
    }
}
