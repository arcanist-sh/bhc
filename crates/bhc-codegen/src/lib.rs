//! Code generation backend for BHC.
//!
//! This crate provides the infrastructure for generating native code from
//! the compiler's intermediate representations. The primary backend is LLVM,
//! but the architecture supports multiple backends.
//!
//! # Overview
//!
//! Code generation in BHC follows this pipeline:
//!
//! ```text
//! Loop IR ──▶ Backend IR ──▶ Object Code
//!                 │
//!                 ├──▶ LLVM IR ──▶ LLVM Backend
//!                 │
//!                 └──▶ (Future: Cranelift, etc.)
//! ```
//!
//! # Backend Architecture
//!
//! The codegen system is designed around traits that abstract over different
//! backends:
//!
//! - [`CodegenBackend`]: The main trait for code generation backends
//! - [`CodegenContext`]: Context holding state during code generation
//! - [`CodegenModule`]: A compilation unit in the backend's representation
//!
//! # LLVM Backend
//!
//! The LLVM backend (when enabled) provides:
//!
//! - Full optimization pipeline integration
//! - Target-specific code generation
//! - Debug information generation
//! - Link-time optimization (LTO) support

#![warn(missing_docs)]

use bhc_session::{DebugInfo, OptLevel, OutputType};
use bhc_target::TargetSpec;
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during code generation.
#[derive(Debug, Error)]
pub enum CodegenError {
    /// The requested backend is not available.
    #[error("backend not available: {0}")]
    BackendNotAvailable(String),

    /// Target is not supported by the backend.
    #[error("unsupported target: {0}")]
    UnsupportedTarget(String),

    /// LLVM-specific error.
    #[error("LLVM error: {0}")]
    LlvmError(String),

    /// Failed to write output file.
    #[error("failed to write output: {path}")]
    OutputError {
        /// The path that could not be written.
        path: String,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },

    /// Internal code generation error.
    #[error("internal codegen error: {0}")]
    Internal(String),
}

/// Result type for code generation operations.
pub type CodegenResult<T> = Result<T, CodegenError>;

/// Code generation configuration.
#[derive(Clone, Debug)]
pub struct CodegenConfig {
    /// Target specification.
    pub target: TargetSpec,
    /// Optimization level.
    pub opt_level: OptLevel,
    /// Debug information level.
    pub debug_info: DebugInfo,
    /// Whether to use PIC (position-independent code).
    pub pic: bool,
    /// Whether to generate frame pointers.
    pub frame_pointers: bool,
    /// Whether to enable LTO.
    pub lto: bool,
    /// CPU model to target (e.g., "generic", "native").
    pub cpu: String,
}

impl Default for CodegenConfig {
    fn default() -> Self {
        Self {
            target: bhc_target::host_target(),
            opt_level: OptLevel::Default,
            debug_info: DebugInfo::None,
            pic: false,
            frame_pointers: true,
            lto: false,
            cpu: "generic".to_string(),
        }
    }
}

impl CodegenConfig {
    /// Create a new codegen config for the given target.
    #[must_use]
    pub fn for_target(target: TargetSpec) -> Self {
        Self {
            target,
            ..Self::default()
        }
    }

    /// Set the optimization level.
    #[must_use]
    pub fn with_opt_level(mut self, level: OptLevel) -> Self {
        self.opt_level = level;
        self
    }

    /// Set the debug info level.
    #[must_use]
    pub fn with_debug_info(mut self, level: DebugInfo) -> Self {
        self.debug_info = level;
        self
    }

    /// Enable position-independent code.
    #[must_use]
    pub fn with_pic(mut self, pic: bool) -> Self {
        self.pic = pic;
        self
    }

    /// Enable LTO.
    #[must_use]
    pub fn with_lto(mut self, lto: bool) -> Self {
        self.lto = lto;
        self
    }
}

/// The type of code being generated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodegenOutputType {
    /// Object file.
    Object,
    /// Assembly file.
    Assembly,
    /// LLVM IR (text).
    LlvmIr,
    /// LLVM bitcode.
    LlvmBitcode,
}

impl From<OutputType> for CodegenOutputType {
    fn from(value: OutputType) -> Self {
        match value {
            OutputType::Assembly => Self::Assembly,
            OutputType::LlvmIr => Self::LlvmIr,
            OutputType::LlvmBitcode => Self::LlvmBitcode,
            _ => Self::Object,
        }
    }
}

/// A module being compiled in the backend.
pub trait CodegenModule: Send {
    /// Get the module name.
    fn name(&self) -> &str;

    /// Verify the module is well-formed.
    fn verify(&self) -> CodegenResult<()>;

    /// Optimize the module.
    fn optimize(&mut self, level: OptLevel) -> CodegenResult<()>;

    /// Write the module to a file.
    fn write_to_file(&self, path: &Path, output_type: CodegenOutputType) -> CodegenResult<()>;

    /// Get the module as LLVM IR text (if supported).
    fn as_llvm_ir(&self) -> CodegenResult<String>;
}

/// Context for code generation.
pub trait CodegenContext: Send + Sync {
    /// The module type for this context.
    type Module: CodegenModule;

    /// Create a new module.
    fn create_module(&self, name: &str) -> CodegenResult<Self::Module>;

    /// Get the target specification.
    fn target(&self) -> &TargetSpec;

    /// Get the codegen configuration.
    fn config(&self) -> &CodegenConfig;
}

/// A code generation backend.
pub trait CodegenBackend: Send + Sync {
    /// The context type for this backend.
    type Context: CodegenContext;

    /// Get the backend name.
    fn name(&self) -> &'static str;

    /// Check if this backend supports the given target.
    fn supports_target(&self, target: &TargetSpec) -> bool;

    /// Create a codegen context with the given configuration.
    fn create_context(&self, config: CodegenConfig) -> CodegenResult<Self::Context>;
}

/// A placeholder LLVM backend (actual implementation requires LLVM bindings).
pub struct LlvmBackend {
    /// Whether LLVM is available.
    available: bool,
}

impl LlvmBackend {
    /// Create a new LLVM backend.
    ///
    /// Returns `None` if LLVM is not available.
    #[must_use]
    pub fn new() -> Option<Self> {
        // In the real implementation, this would check for LLVM availability
        Some(Self { available: true })
    }

    /// Check if LLVM is available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.available
    }
}

/// Placeholder LLVM context.
pub struct LlvmContext {
    config: CodegenConfig,
}

/// Placeholder LLVM module.
pub struct LlvmModule {
    name: String,
    ir: String,
}

impl CodegenModule for LlvmModule {
    fn name(&self) -> &str {
        &self.name
    }

    fn verify(&self) -> CodegenResult<()> {
        // Placeholder: would verify LLVM module
        Ok(())
    }

    fn optimize(&mut self, _level: OptLevel) -> CodegenResult<()> {
        // Placeholder: would run LLVM optimization passes
        Ok(())
    }

    fn write_to_file(&self, path: &Path, output_type: CodegenOutputType) -> CodegenResult<()> {
        use std::fs;

        let content = match output_type {
            CodegenOutputType::LlvmIr => self.ir.clone(),
            CodegenOutputType::LlvmBitcode => {
                // Placeholder: would emit bitcode
                "placeholder bitcode".to_string()
            }
            CodegenOutputType::Assembly => {
                // Placeholder: would emit assembly
                "; placeholder assembly\n".to_string()
            }
            CodegenOutputType::Object => {
                // Placeholder: would emit object file
                return Err(CodegenError::Internal(
                    "object file emission not implemented in placeholder".to_string(),
                ));
            }
        };

        fs::write(path, content).map_err(|e| CodegenError::OutputError {
            path: path.display().to_string(),
            source: e,
        })
    }

    fn as_llvm_ir(&self) -> CodegenResult<String> {
        Ok(self.ir.clone())
    }
}

impl CodegenContext for LlvmContext {
    type Module = LlvmModule;

    fn create_module(&self, name: &str) -> CodegenResult<LlvmModule> {
        Ok(LlvmModule {
            name: name.to_string(),
            ir: format!(
                "; ModuleID = '{}'\n\
                 source_filename = \"{}\"\n\
                 target datalayout = \"{}\"\n\
                 target triple = \"{}\"\n",
                name,
                name,
                self.config.target.data_layout,
                self.config.target.triple()
            ),
        })
    }

    fn target(&self) -> &TargetSpec {
        &self.config.target
    }

    fn config(&self) -> &CodegenConfig {
        &self.config
    }
}

impl CodegenBackend for LlvmBackend {
    type Context = LlvmContext;

    fn name(&self) -> &'static str {
        "llvm"
    }

    fn supports_target(&self, _target: &TargetSpec) -> bool {
        // LLVM supports all our targets
        true
    }

    fn create_context(&self, config: CodegenConfig) -> CodegenResult<LlvmContext> {
        if !self.available {
            return Err(CodegenError::BackendNotAvailable("LLVM".to_string()));
        }
        Ok(LlvmContext { config })
    }
}

/// Builder for LLVM IR (simplified placeholder).
pub struct IrBuilder {
    instructions: Vec<String>,
}

impl IrBuilder {
    /// Create a new IR builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
        }
    }

    /// Add a function definition.
    pub fn define_function(&mut self, name: &str, ret_type: &str, params: &str) {
        self.instructions.push(format!(
            "define {} @{}({}) {{\nentry:",
            ret_type, name, params
        ));
    }

    /// Add a return instruction.
    pub fn build_ret(&mut self, value: &str) {
        self.instructions.push(format!("  ret {}", value));
        self.instructions.push("}".to_string());
    }

    /// Build the IR string.
    #[must_use]
    pub fn build(&self) -> String {
        self.instructions.join("\n")
    }
}

impl Default for IrBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Type layout information for code generation.
#[derive(Clone, Debug)]
pub struct TypeLayout {
    /// Size in bytes.
    pub size: u64,
    /// Alignment in bytes.
    pub alignment: u64,
}

impl TypeLayout {
    /// Layout for a pointer on the given target.
    #[must_use]
    pub fn pointer(target: &TargetSpec) -> Self {
        let width = target.pointer_width() as u64;
        Self {
            size: width,
            alignment: width,
        }
    }

    /// Layout for an i64.
    #[must_use]
    pub const fn i64() -> Self {
        Self {
            size: 8,
            alignment: 8,
        }
    }

    /// Layout for an f64.
    #[must_use]
    pub const fn f64() -> Self {
        Self {
            size: 8,
            alignment: 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_config() {
        let config = CodegenConfig::default()
            .with_opt_level(OptLevel::Aggressive)
            .with_pic(true);

        assert_eq!(config.opt_level, OptLevel::Aggressive);
        assert!(config.pic);
    }

    #[test]
    fn test_llvm_backend_creation() {
        let backend = LlvmBackend::new().unwrap();
        assert_eq!(backend.name(), "llvm");
        assert!(backend.is_available());
    }

    #[test]
    fn test_module_creation() {
        let backend = LlvmBackend::new().unwrap();
        let ctx = backend.create_context(CodegenConfig::default()).unwrap();
        let module = ctx.create_module("test").unwrap();

        assert_eq!(module.name(), "test");
        assert!(module.as_llvm_ir().unwrap().contains("ModuleID = 'test'"));
    }

    #[test]
    fn test_ir_builder() {
        let mut builder = IrBuilder::new();
        builder.define_function("main", "i32", "");
        builder.build_ret("i32 0");

        let ir = builder.build();
        assert!(ir.contains("define i32 @main()"));
        assert!(ir.contains("ret i32 0"));
    }
}
