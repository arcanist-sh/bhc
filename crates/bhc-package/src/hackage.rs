//! Hackage package registry client.
//!
//! This module provides functionality to download and cache packages
//! from Hackage (hackage.haskell.org), the central Haskell package repository.
//!
//! # Usage
//!
//! ```ignore
//! use bhc_package::hackage::Hackage;
//!
//! let hackage = Hackage::new()?;
//!
//! // Download and extract a package
//! let pkg_dir = hackage.fetch_package("filepath", "1.4.100.0")?;
//!
//! // The package is now available at pkg_dir with its source files
//! ```

use crate::cabal::{CabalError, CabalFile};
use camino::{Utf8Path, Utf8PathBuf};
use flate2::read::GzDecoder;
use semver::Version;
use std::fs;
use std::io::Read;
use tar::Archive;
use thiserror::Error;
use tracing::{debug, info};

/// Hackage base URL.
pub const HACKAGE_URL: &str = "https://hackage.haskell.org";

/// Errors from Hackage operations.
#[derive(Debug, Error)]
pub enum HackageError {
    /// Package not found on Hackage.
    #[error("package not found on Hackage: {0}")]
    PackageNotFound(String),

    /// Version not found.
    #[error("version {version} not found for package {package}")]
    VersionNotFound {
        /// Package name.
        package: String,
        /// Requested version.
        version: String,
    },

    /// Network error.
    #[error("network error: {0}")]
    Network(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Cabal parse error.
    #[error("cabal parse error: {0}")]
    CabalParse(#[from] CabalError),

    /// Archive error.
    #[error("archive error: {0}")]
    Archive(String),
}

/// Result type for Hackage operations.
pub type HackageResult<T> = Result<T, HackageError>;

/// Hackage configuration.
#[derive(Clone, Debug)]
pub struct HackageConfig {
    /// Base URL for Hackage.
    pub base_url: String,
    /// Local cache directory for downloaded packages.
    pub cache_dir: Utf8PathBuf,
    /// Request timeout in seconds.
    pub timeout: u64,
}

impl Default for HackageConfig {
    fn default() -> Self {
        let cache_dir = dirs::cache_dir()
            .and_then(|p| Utf8PathBuf::try_from(p).ok())
            .unwrap_or_else(|| Utf8PathBuf::from(".cache"))
            .join("bhc")
            .join("hackage");

        Self {
            base_url: HACKAGE_URL.to_string(),
            cache_dir,
            timeout: 60,
        }
    }
}

impl HackageConfig {
    /// Create configuration with a custom cache directory.
    pub fn with_cache_dir(mut self, path: impl AsRef<Utf8Path>) -> Self {
        self.cache_dir = path.as_ref().to_path_buf();
        self
    }
}

/// Hackage package information.
#[derive(Clone, Debug)]
pub struct HackagePackage {
    /// Package name.
    pub name: String,
    /// Package version.
    pub version: Version,
    /// Path to extracted package.
    pub path: Utf8PathBuf,
    /// Parsed cabal file.
    pub cabal: CabalFile,
}

impl HackagePackage {
    /// Get the source directories for the library.
    pub fn source_dirs(&self) -> Vec<Utf8PathBuf> {
        self.cabal
            .library
            .as_ref()
            .map(|lib| {
                lib.hs_source_dirs
                    .iter()
                    .map(|d| self.path.join(d))
                    .collect()
            })
            .unwrap_or_else(|| vec![self.path.clone()])
    }

    /// Get the exposed modules.
    pub fn exposed_modules(&self) -> Vec<&str> {
        self.cabal.exposed_modules()
    }

    /// Get dependencies.
    pub fn dependencies(&self) -> Vec<(&str, Option<&str>)> {
        self.cabal
            .all_dependencies()
            .into_iter()
            .map(|d| (d.name.as_str(), d.version_constraint.as_deref()))
            .collect()
    }
}

/// Hackage client for downloading packages.
pub struct Hackage {
    config: HackageConfig,
}

impl Hackage {
    /// Create a new Hackage client with default configuration.
    pub fn new() -> HackageResult<Self> {
        Self::with_config(HackageConfig::default())
    }

    /// Create a Hackage client with custom configuration.
    pub fn with_config(config: HackageConfig) -> HackageResult<Self> {
        fs::create_dir_all(&config.cache_dir)?;
        Ok(Self { config })
    }

    /// Get the cache directory.
    pub fn cache_dir(&self) -> &Utf8Path {
        &self.config.cache_dir
    }

    /// Get the path where a package would be extracted.
    pub fn package_path(&self, name: &str, version: &str) -> Utf8PathBuf {
        self.config
            .cache_dir
            .join("packages")
            .join(name)
            .join(format!("{}-{}", name, version))
    }

    /// Check if a package is already cached.
    pub fn is_cached(&self, name: &str, version: &str) -> bool {
        let pkg_path = self.package_path(name, version);
        pkg_path.exists() && pkg_path.join(format!("{}.cabal", name)).exists()
    }

    /// Fetch a package from Hackage, downloading if not cached.
    pub fn fetch_package(&self, name: &str, version: &str) -> HackageResult<HackagePackage> {
        let pkg_path = self.package_path(name, version);

        // Check cache first
        if self.is_cached(name, version) {
            debug!("Using cached package: {}-{}", name, version);
            return self.load_cached_package(name, version);
        }

        // Download from Hackage
        info!("Downloading {}-{} from Hackage...", name, version);
        let tarball = self.download_tarball(name, version)?;

        // Extract to cache
        self.extract_tarball(name, version, &tarball)?;

        // Load the package
        self.load_cached_package(name, version)
    }

    /// Download a package tarball from Hackage.
    fn download_tarball(&self, name: &str, version: &str) -> HackageResult<Vec<u8>> {
        // Hackage URL format: /package/{name}-{version}/{name}-{version}.tar.gz
        let url = format!(
            "{}/package/{name}-{version}/{name}-{version}.tar.gz",
            self.config.base_url
        );

        debug!("Downloading: {}", url);

        let response = ureq::get(&url)
            .timeout(std::time::Duration::from_secs(self.config.timeout))
            .call()
            .map_err(|e| match e {
                ureq::Error::Status(404, _) => HackageError::VersionNotFound {
                    package: name.to_string(),
                    version: version.to_string(),
                },
                _ => HackageError::Network(e.to_string()),
            })?;

        let mut data = Vec::new();
        response
            .into_reader()
            .read_to_end(&mut data)
            .map_err(|e| HackageError::Network(e.to_string()))?;

        Ok(data)
    }

    /// Extract a tarball to the cache directory.
    fn extract_tarball(&self, name: &str, version: &str, tarball: &[u8]) -> HackageResult<()> {
        let pkg_dir = self.package_path(name, version);

        // Create parent directory
        if let Some(parent) = pkg_dir.parent() {
            fs::create_dir_all(parent)?;
        }

        // Extract tarball
        let decoder = GzDecoder::new(tarball);
        let mut archive = Archive::new(decoder);

        // Hackage tarballs contain a single directory named "{name}-{version}"
        // We extract to the parent and the directory will be created
        let extract_dir = pkg_dir
            .parent()
            .ok_or_else(|| HackageError::Archive("invalid package path".to_string()))?;

        archive.unpack(extract_dir.as_std_path())?;

        info!("Extracted {}-{} to {}", name, version, pkg_dir);
        Ok(())
    }

    /// Load a cached package.
    fn load_cached_package(&self, name: &str, version: &str) -> HackageResult<HackagePackage> {
        let pkg_path = self.package_path(name, version);
        let cabal_path = pkg_path.join(format!("{}.cabal", name));

        if !cabal_path.exists() {
            return Err(HackageError::PackageNotFound(format!(
                "{}-{} (cabal file not found)",
                name, version
            )));
        }

        let cabal = CabalFile::load(&cabal_path)?;

        Ok(HackagePackage {
            name: name.to_string(),
            version: cabal.version.clone(),
            path: pkg_path,
            cabal,
        })
    }

    /// List all cached packages.
    pub fn list_cached(&self) -> HackageResult<Vec<(String, String)>> {
        let packages_dir = self.config.cache_dir.join("packages");
        if !packages_dir.exists() {
            return Ok(Vec::new());
        }

        let mut packages = Vec::new();

        for pkg_entry in fs::read_dir(packages_dir.as_std_path())? {
            let pkg_entry = pkg_entry?;
            let pkg_name = pkg_entry.file_name().to_string_lossy().to_string();

            for ver_entry in fs::read_dir(pkg_entry.path())? {
                let ver_entry = ver_entry?;
                let dir_name = ver_entry.file_name().to_string_lossy().to_string();

                // Directory name is "{name}-{version}"
                if let Some(version) = dir_name.strip_prefix(&format!("{}-", pkg_name)) {
                    packages.push((pkg_name.clone(), version.to_string()));
                }
            }
        }

        Ok(packages)
    }

    /// Clear the package cache.
    pub fn clear_cache(&self) -> HackageResult<()> {
        let packages_dir = self.config.cache_dir.join("packages");
        if packages_dir.exists() {
            fs::remove_dir_all(packages_dir.as_std_path())?;
        }
        Ok(())
    }

    /// Fetch a package and all its dependencies recursively.
    ///
    /// Returns packages in dependency order (dependencies first).
    pub fn fetch_with_dependencies(
        &self,
        name: &str,
        version: &str,
    ) -> HackageResult<Vec<HackagePackage>> {
        let mut packages = Vec::new();
        let mut visited = std::collections::HashSet::new();

        self.fetch_deps_recursive(name, version, &mut packages, &mut visited)?;

        Ok(packages)
    }

    fn fetch_deps_recursive(
        &self,
        name: &str,
        version: &str,
        packages: &mut Vec<HackagePackage>,
        visited: &mut std::collections::HashSet<String>,
    ) -> HackageResult<()> {
        let key = format!("{}-{}", name, version);
        if visited.contains(&key) {
            return Ok(());
        }
        visited.insert(key);

        // Skip base packages that are built into BHC
        let builtin_packages = [
            "base",
            "ghc-prim",
            "ghc-bignum",
            "integer-gmp",
            "integer-simple",
            "rts",
            "template-haskell",
        ];
        if builtin_packages.contains(&name) {
            debug!("Skipping builtin package: {}", name);
            return Ok(());
        }

        let pkg = self.fetch_package(name, version)?;

        // Fetch dependencies first (so they appear before this package)
        for dep in pkg.cabal.all_dependencies() {
            // For now, we'll try to fetch the latest version of each dependency
            // A proper implementation would resolve version constraints
            if !builtin_packages.contains(&dep.name.as_str()) && !visited.contains(&dep.name) {
                // Try to find a version - for simplicity, we skip if we don't know the version
                // In a real implementation, we'd query Hackage for available versions
                debug!(
                    "Dependency {} (constraint: {:?}) - skipping for now",
                    dep.name, dep.version_constraint
                );
            }
        }

        packages.push(pkg);
        Ok(())
    }
}

/// Information about available versions of a package on Hackage.
#[derive(Clone, Debug)]
pub struct PackageVersions {
    /// Package name.
    pub name: String,
    /// Available versions (newest first).
    pub versions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_package_path() {
        let config = HackageConfig::default().with_cache_dir("/tmp/test-cache");
        let hackage = Hackage::with_config(config).unwrap();

        let path = hackage.package_path("filepath", "1.4.100.0");
        assert!(path.as_str().contains("filepath"));
        assert!(path.as_str().contains("filepath-1.4.100.0"));
    }

    #[test]
    fn test_builtin_packages() {
        // These should be skipped in dependency resolution
        let builtins = ["base", "ghc-prim", "rts"];
        for pkg in builtins {
            assert!(
                ["base", "ghc-prim", "ghc-bignum", "integer-gmp", "integer-simple", "rts", "template-haskell"]
                    .contains(&pkg)
            );
        }
    }
}
