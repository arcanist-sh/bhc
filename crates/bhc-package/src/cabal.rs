//! Minimal Cabal file parser for BHC.
//!
//! This module provides a simplified parser for Haskell `.cabal` files,
//! extracting only the fields needed for BHC compilation:
//!
//! - Package name and version
//! - Library exposed modules and source directories
//! - Build dependencies
//!
//! This is NOT a full cabal parser - it ignores conditionals, flags,
//! and many optional fields. It's designed to work with simple packages.

use camino::{Utf8Path, Utf8PathBuf};
use semver::Version;
use thiserror::Error;

/// Errors that can occur during cabal parsing.
#[derive(Debug, Error)]
pub enum CabalError {
    /// IO error reading the file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Missing required field.
    #[error("missing required field: {0}")]
    MissingField(&'static str),

    /// Invalid version string.
    #[error("invalid version: {0}")]
    InvalidVersion(String),

    /// Parse error.
    #[error("parse error at line {line}: {message}")]
    Parse {
        /// Line number where the error occurred.
        line: usize,
        /// Error message.
        message: String,
    },
}

/// Result type for cabal operations.
pub type CabalResult<T> = Result<T, CabalError>;

/// A parsed .cabal file.
#[derive(Clone, Debug)]
pub struct CabalFile {
    /// Package name.
    pub name: String,
    /// Package version.
    pub version: Version,
    /// Package synopsis/description.
    pub synopsis: Option<String>,
    /// License.
    pub license: Option<String>,
    /// Library configuration (if present).
    pub library: Option<CabalLibrary>,
    /// Executable configurations.
    pub executables: Vec<CabalExecutable>,
    /// Top-level build dependencies.
    pub build_depends: Vec<CabalDependency>,
}

/// Library stanza from a cabal file.
#[derive(Clone, Debug, Default)]
pub struct CabalLibrary {
    /// Exposed modules.
    pub exposed_modules: Vec<String>,
    /// Other (non-exposed) modules.
    pub other_modules: Vec<String>,
    /// Source directories.
    pub hs_source_dirs: Vec<Utf8PathBuf>,
    /// Build dependencies.
    pub build_depends: Vec<CabalDependency>,
}

/// Executable stanza from a cabal file.
#[derive(Clone, Debug)]
pub struct CabalExecutable {
    /// Executable name.
    pub name: String,
    /// Main module file.
    pub main_is: String,
    /// Source directories.
    pub hs_source_dirs: Vec<Utf8PathBuf>,
    /// Build dependencies.
    pub build_depends: Vec<CabalDependency>,
}

/// A dependency specification.
#[derive(Clone, Debug)]
pub struct CabalDependency {
    /// Package name.
    pub name: String,
    /// Version constraint (unparsed, e.g., ">=1.0 && <2.0").
    pub version_constraint: Option<String>,
}

impl CabalFile {
    /// Parse a cabal file from a path.
    pub fn load(path: impl AsRef<Utf8Path>) -> CabalResult<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        Self::parse(&content)
    }

    /// Parse cabal file content.
    pub fn parse(content: &str) -> CabalResult<Self> {
        let mut parser = CabalParser::new(content);
        parser.parse()
    }

    /// Get all source directories for the library.
    pub fn library_source_dirs(&self) -> Vec<&Utf8Path> {
        self.library
            .as_ref()
            .map(|lib| {
                if lib.hs_source_dirs.is_empty() {
                    vec![Utf8Path::new(".")]
                } else {
                    lib.hs_source_dirs.iter().map(|p| p.as_path()).collect()
                }
            })
            .unwrap_or_default()
    }

    /// Get all exposed modules.
    pub fn exposed_modules(&self) -> Vec<&str> {
        self.library
            .as_ref()
            .map(|lib| lib.exposed_modules.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get all build dependencies (library + top-level).
    pub fn all_dependencies(&self) -> Vec<&CabalDependency> {
        let mut deps: Vec<&CabalDependency> = self.build_depends.iter().collect();
        if let Some(ref lib) = self.library {
            deps.extend(lib.build_depends.iter());
        }
        deps
    }
}

/// Internal parser state.
struct CabalParser<'a> {
    lines: Vec<&'a str>,
    pos: usize,
}

impl<'a> CabalParser<'a> {
    fn new(content: &'a str) -> Self {
        Self {
            lines: content.lines().collect(),
            pos: 0,
        }
    }

    fn parse(&mut self) -> CabalResult<CabalFile> {
        let mut name = None;
        let mut version = None;
        let mut synopsis = None;
        let mut license = None;
        let mut library = None;
        let mut executables = Vec::new();
        let mut build_depends = Vec::new();

        while self.pos < self.lines.len() {
            let line = self.lines[self.pos];
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with("--") {
                self.pos += 1;
                continue;
            }

            // Check for stanza headers
            let lower = trimmed.to_lowercase();
            if lower == "library" {
                self.pos += 1;
                library = Some(self.parse_library_stanza()?);
                continue;
            } else if lower.starts_with("executable ") {
                let exec_name = trimmed[11..].trim().to_string();
                self.pos += 1;
                executables.push(self.parse_executable_stanza(exec_name)?);
                continue;
            } else if lower.starts_with("test-suite ")
                || lower.starts_with("benchmark ")
                || lower.starts_with("source-repository ")
                || lower.starts_with("flag ")
                || lower.starts_with("common ")
            {
                // Skip these stanzas
                self.pos += 1;
                self.skip_stanza();
                continue;
            }

            // Parse top-level field
            if let Some((key, value)) = self.parse_field(line) {
                match key.to_lowercase().as_str() {
                    "name" => name = Some(value.to_string()),
                    "version" => version = Some(value.to_string()),
                    "synopsis" => synopsis = Some(value.to_string()),
                    "license" => license = Some(value.to_string()),
                    "build-depends" => {
                        build_depends.extend(self.parse_dependencies(value)?);
                    }
                    _ => {} // Ignore other fields
                }
            }

            self.pos += 1;
        }

        let name = name.ok_or(CabalError::MissingField("name"))?;
        let version_str = version.ok_or(CabalError::MissingField("version"))?;
        let version = parse_version(&version_str)?;

        Ok(CabalFile {
            name,
            version,
            synopsis,
            license,
            library,
            executables,
            build_depends,
        })
    }

    fn parse_library_stanza(&mut self) -> CabalResult<CabalLibrary> {
        let mut lib = CabalLibrary::default();
        let base_indent = self.get_indent();

        while self.pos < self.lines.len() {
            let line = self.lines[self.pos];
            let current_indent = self.get_line_indent(line);

            // Check if we've left the stanza
            if !line.trim().is_empty() && current_indent < base_indent {
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("--") {
                self.pos += 1;
                continue;
            }

            // Check for conditional blocks and skip them
            let lower = trimmed.to_lowercase();
            if lower.starts_with("if ") || lower.starts_with("else") {
                self.pos += 1;
                self.skip_conditional_block(current_indent);
                continue;
            }

            if let Some((key, value)) = self.parse_field(line) {
                match key.to_lowercase().as_str() {
                    "exposed-modules" => {
                        lib.exposed_modules.extend(self.parse_module_list(value));
                    }
                    "other-modules" => {
                        lib.other_modules.extend(self.parse_module_list(value));
                    }
                    "hs-source-dirs" => {
                        lib.hs_source_dirs
                            .extend(self.parse_path_list(value));
                    }
                    "build-depends" => {
                        lib.build_depends.extend(self.parse_dependencies(value)?);
                    }
                    _ => {} // Ignore other fields
                }
            }

            self.pos += 1;
        }

        // Default source dir if none specified
        if lib.hs_source_dirs.is_empty() {
            lib.hs_source_dirs.push(Utf8PathBuf::from("."));
        }

        Ok(lib)
    }

    fn parse_executable_stanza(&mut self, name: String) -> CabalResult<CabalExecutable> {
        let mut exe = CabalExecutable {
            name,
            main_is: "Main.hs".to_string(),
            hs_source_dirs: Vec::new(),
            build_depends: Vec::new(),
        };
        let base_indent = self.get_indent();

        while self.pos < self.lines.len() {
            let line = self.lines[self.pos];
            let current_indent = self.get_line_indent(line);

            if !line.trim().is_empty() && current_indent < base_indent {
                break;
            }

            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("--") {
                self.pos += 1;
                continue;
            }

            // Skip conditionals
            let lower = trimmed.to_lowercase();
            if lower.starts_with("if ") || lower.starts_with("else") {
                self.pos += 1;
                self.skip_conditional_block(current_indent);
                continue;
            }

            if let Some((key, value)) = self.parse_field(line) {
                match key.to_lowercase().as_str() {
                    "main-is" => exe.main_is = value.to_string(),
                    "hs-source-dirs" => {
                        exe.hs_source_dirs.extend(self.parse_path_list(value));
                    }
                    "build-depends" => {
                        exe.build_depends.extend(self.parse_dependencies(value)?);
                    }
                    _ => {}
                }
            }

            self.pos += 1;
        }

        if exe.hs_source_dirs.is_empty() {
            exe.hs_source_dirs.push(Utf8PathBuf::from("."));
        }

        Ok(exe)
    }

    fn skip_stanza(&mut self) {
        let base_indent = self.get_indent();
        while self.pos < self.lines.len() {
            let line = self.lines[self.pos];
            let current_indent = self.get_line_indent(line);
            if !line.trim().is_empty() && current_indent < base_indent {
                break;
            }
            self.pos += 1;
        }
    }

    fn skip_conditional_block(&mut self, base_indent: usize) {
        while self.pos < self.lines.len() {
            let line = self.lines[self.pos];
            let current_indent = self.get_line_indent(line);
            if !line.trim().is_empty() && current_indent <= base_indent {
                break;
            }
            self.pos += 1;
        }
    }

    fn get_indent(&self) -> usize {
        if self.pos < self.lines.len() {
            self.get_line_indent(self.lines[self.pos])
        } else {
            0
        }
    }

    fn get_line_indent(&self, line: &str) -> usize {
        line.len() - line.trim_start().len()
    }

    fn parse_field<'b>(&self, line: &'b str) -> Option<(&'b str, &'b str)> {
        let trimmed = line.trim();
        if let Some(colon_pos) = trimmed.find(':') {
            let key = trimmed[..colon_pos].trim();
            let value = trimmed[colon_pos + 1..].trim();
            Some((key, value))
        } else {
            None
        }
    }

    fn parse_module_list(&mut self, first_value: &str) -> Vec<String> {
        let mut modules = Vec::new();

        // Parse first line
        for part in first_value.split(',') {
            let module = part.trim();
            if !module.is_empty() {
                modules.push(module.to_string());
            }
        }

        // Check for continuation lines
        let base_indent = self.get_indent();
        while self.pos + 1 < self.lines.len() {
            let next_line = self.lines[self.pos + 1];
            let next_indent = self.get_line_indent(next_line);
            let trimmed = next_line.trim();

            // Continuation must be indented and not a new field
            if next_indent > base_indent && !trimmed.contains(':') && !trimmed.starts_with("--") {
                self.pos += 1;
                for part in trimmed.split(',') {
                    let module = part.trim();
                    if !module.is_empty() {
                        modules.push(module.to_string());
                    }
                }
            } else {
                break;
            }
        }

        modules
    }

    fn parse_path_list(&mut self, first_value: &str) -> Vec<Utf8PathBuf> {
        let mut paths = Vec::new();

        for part in first_value.split(',') {
            let path = part.trim();
            if !path.is_empty() {
                paths.push(Utf8PathBuf::from(path));
            }
        }

        // Check for continuation lines
        let base_indent = self.get_indent();
        while self.pos + 1 < self.lines.len() {
            let next_line = self.lines[self.pos + 1];
            let next_indent = self.get_line_indent(next_line);
            let trimmed = next_line.trim();

            if next_indent > base_indent && !trimmed.contains(':') && !trimmed.starts_with("--") {
                self.pos += 1;
                for part in trimmed.split(',') {
                    let path = part.trim();
                    if !path.is_empty() {
                        paths.push(Utf8PathBuf::from(path));
                    }
                }
            } else {
                break;
            }
        }

        paths
    }

    fn parse_dependencies(&mut self, first_value: &str) -> CabalResult<Vec<CabalDependency>> {
        let mut deps = Vec::new();

        // Collect all dependency text (may span multiple lines)
        let mut dep_text = first_value.to_string();

        let base_indent = self.get_indent();
        while self.pos + 1 < self.lines.len() {
            let next_line = self.lines[self.pos + 1];
            let next_indent = self.get_line_indent(next_line);
            let trimmed = next_line.trim();

            if next_indent > base_indent && !trimmed.contains(':') && !trimmed.starts_with("--") {
                self.pos += 1;
                dep_text.push(' ');
                dep_text.push_str(trimmed);
            } else {
                break;
            }
        }

        // Parse dependencies (comma-separated)
        for part in dep_text.split(',') {
            let dep_str = part.trim();
            if dep_str.is_empty() {
                continue;
            }

            // Parse "package-name >=1.0 && <2.0" format
            let (name, constraint) = parse_dependency_spec(dep_str);
            deps.push(CabalDependency {
                name: name.to_string(),
                version_constraint: constraint.map(|s| s.to_string()),
            });
        }

        Ok(deps)
    }
}

/// Parse a dependency specification like "base >=4.7 && <5".
fn parse_dependency_spec(spec: &str) -> (&str, Option<&str>) {
    let spec = spec.trim();

    // Find where the version constraint starts
    // It typically starts with a comparison operator or version number
    let constraint_start = spec
        .find(|c: char| c == '>' || c == '<' || c == '=' || c == '^')
        .or_else(|| {
            // Also check for version numbers after space
            spec.find(' ').and_then(|space_pos| {
                let after_space = &spec[space_pos + 1..];
                if after_space
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false)
                {
                    Some(space_pos)
                } else {
                    None
                }
            })
        });

    match constraint_start {
        Some(pos) => {
            let name = spec[..pos].trim();
            let constraint = spec[pos..].trim();
            (name, Some(constraint))
        }
        None => (spec, None),
    }
}

/// Parse a version string, handling common cabal version formats.
fn parse_version(version_str: &str) -> CabalResult<Version> {
    let cleaned = version_str.trim();

    // Handle versions with more than 3 parts (e.g., "1.2.3.4")
    let parts: Vec<&str> = cleaned.split('.').collect();
    let normalized = if parts.len() > 3 {
        // Take only first 3 parts for semver compatibility
        parts[..3].join(".")
    } else if parts.len() == 2 {
        // Add .0 for two-part versions
        format!("{}.0", cleaned)
    } else if parts.len() == 1 {
        // Add .0.0 for single-part versions
        format!("{}.0.0", cleaned)
    } else {
        cleaned.to_string()
    };

    Version::parse(&normalized).map_err(|_| CabalError::InvalidVersion(version_str.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CABAL: &str = r#"
name:                filepath
version:             1.4.100.0
synopsis:            Library for manipulating FilePaths in a cross platform way.
license:             BSD3

library
  exposed-modules:
    System.FilePath
    System.FilePath.Posix
    System.FilePath.Windows
  hs-source-dirs:   .
  build-depends:
    base >= 4.9 && < 5

executable filepath-tests
  main-is:          Main.hs
  hs-source-dirs:   tests
  build-depends:    base, filepath
"#;

    #[test]
    fn test_parse_cabal() {
        let cabal = CabalFile::parse(SAMPLE_CABAL).unwrap();
        assert_eq!(cabal.name, "filepath");
        assert_eq!(cabal.version.major, 1);
        assert_eq!(cabal.version.minor, 4);
        assert_eq!(cabal.synopsis, Some("Library for manipulating FilePaths in a cross platform way.".to_string()));
    }

    #[test]
    fn test_parse_library() {
        let cabal = CabalFile::parse(SAMPLE_CABAL).unwrap();
        let lib = cabal.library.as_ref().unwrap();

        assert_eq!(lib.exposed_modules.len(), 3);
        assert!(lib.exposed_modules.contains(&"System.FilePath".to_string()));
        assert!(lib.exposed_modules.contains(&"System.FilePath.Posix".to_string()));
    }

    #[test]
    fn test_parse_dependencies() {
        let cabal = CabalFile::parse(SAMPLE_CABAL).unwrap();
        let lib = cabal.library.as_ref().unwrap();

        assert!(!lib.build_depends.is_empty());
        let base_dep = lib.build_depends.iter().find(|d| d.name == "base").unwrap();
        assert!(base_dep.version_constraint.is_some());
    }

    #[test]
    fn test_parse_executable() {
        let cabal = CabalFile::parse(SAMPLE_CABAL).unwrap();
        assert_eq!(cabal.executables.len(), 1);

        let exe = &cabal.executables[0];
        assert_eq!(exe.name, "filepath-tests");
        assert_eq!(exe.main_is, "Main.hs");
    }

    #[test]
    fn test_dependency_spec_parsing() {
        let (name, constraint) = parse_dependency_spec("base >=4.7 && <5");
        assert_eq!(name, "base");
        assert_eq!(constraint, Some(">=4.7 && <5"));

        let (name, constraint) = parse_dependency_spec("containers");
        assert_eq!(name, "containers");
        assert_eq!(constraint, None);

        let (name, constraint) = parse_dependency_spec("text ^>=1.2");
        assert_eq!(name, "text");
        assert_eq!(constraint, Some("^>=1.2"));
    }

    #[test]
    fn test_version_parsing() {
        // Standard semver
        let v = parse_version("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);

        // Four-part version (common in Haskell)
        let v = parse_version("1.4.100.0").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 4);
        assert_eq!(v.patch, 100);

        // Two-part version
        let v = parse_version("4.7").unwrap();
        assert_eq!(v.major, 4);
        assert_eq!(v.minor, 7);
        assert_eq!(v.patch, 0);
    }
}
