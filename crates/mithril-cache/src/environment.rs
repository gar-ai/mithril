//! Runtime environment capture for cache compatibility checking.
//!
//! This module captures information about the current runtime environment
//! that affects cache compatibility (Python version, PyTorch version, CUDA, etc.).

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

/// Semantic version with major.minor.patch components.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    /// Optional suffix like "rc1", "dev", etc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
}

impl Version {
    /// Create a new version.
    #[must_use]
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            suffix: None,
        }
    }

    /// Create a version with a suffix.
    #[must_use]
    pub fn with_suffix(major: u32, minor: u32, patch: u32, suffix: impl Into<String>) -> Self {
        Self {
            major,
            minor,
            patch,
            suffix: Some(suffix.into()),
        }
    }

    /// Check if major.minor matches.
    #[must_use]
    pub fn major_minor_matches(&self, other: &Self) -> bool {
        self.major == other.major && self.minor == other.minor
    }

    /// Check if this version is >= another version.
    #[must_use]
    pub fn is_gte(&self, other: &Self) -> bool {
        match self.major.cmp(&other.major) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => match self.minor.cmp(&other.minor) {
                Ordering::Greater => true,
                Ordering::Less => false,
                Ordering::Equal => self.patch >= other.patch,
            },
        }
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(suffix) = &self.suffix {
            write!(f, "{}.{}.{}{}", self.major, self.minor, self.patch, suffix)
        } else {
            write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
        }
    }
}

impl FromStr for Version {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Handle versions like "2.1.0", "3.11.4", "2.1.0+cu118", "2.0.0.dev20230101"
        let s = s.trim();

        // Find where the numeric part ends
        let mut parts_end = s.len();
        let mut found_suffix = false;
        for (i, c) in s.char_indices() {
            if !c.is_ascii_digit() && c != '.' {
                parts_end = i;
                found_suffix = true;
                break;
            }
        }

        let version_part = &s[..parts_end];
        let suffix = if found_suffix && parts_end < s.len() {
            Some(s[parts_end..].to_string())
        } else {
            None
        };

        let parts: Vec<&str> = version_part.split('.').collect();
        if parts.is_empty() || parts.len() > 4 {
            return Err(format!("Invalid version format: {}", s));
        }

        let major = parts[0]
            .parse()
            .map_err(|_| format!("Invalid major version: {}", parts[0]))?;
        let minor = parts
            .get(1)
            .map(|p| p.parse())
            .transpose()
            .map_err(|_| format!("Invalid minor version: {}", parts.get(1).unwrap_or(&"")))?
            .unwrap_or(0);
        let patch = parts
            .get(2)
            .map(|p| p.parse())
            .transpose()
            .map_err(|_| format!("Invalid patch version: {}", parts.get(2).unwrap_or(&"")))?
            .unwrap_or(0);

        Ok(Self {
            major,
            minor,
            patch,
            suffix,
        })
    }
}

/// CUDA compute capability (e.g., "8.0" for A100).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComputeCapability {
    pub major: u32,
    pub minor: u32,
}

impl ComputeCapability {
    #[must_use]
    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Check if this compute capability is >= another.
    #[must_use]
    pub fn is_gte(&self, other: &Self) -> bool {
        match self.major.cmp(&other.major) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => self.minor >= other.minor,
        }
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

impl FromStr for ComputeCapability {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.trim().split('.').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid compute capability: {}", s));
        }

        let major = parts[0]
            .parse()
            .map_err(|_| format!("Invalid major version: {}", parts[0]))?;
        let minor = parts[1]
            .parse()
            .map_err(|_| format!("Invalid minor version: {}", parts[1]))?;

        Ok(Self { major, minor })
    }
}

/// Runtime environment information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Environment {
    /// Python version (e.g., "3.11.4")
    pub python_version: Version,

    /// PyTorch version (e.g., "2.1.0+cu118")
    pub torch_version: Version,

    /// CUDA version if available (e.g., "11.8")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_version: Option<Version>,

    /// CUDA compute capability if available (e.g., "8.0")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_compute: Option<ComputeCapability>,

    /// Triton version if available
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub triton_version: Option<Version>,

    /// Platform identifier (e.g., "linux-x86_64", "darwin-arm64")
    pub platform: String,

    /// CPU instruction set features (e.g., ["avx2", "avx512"])
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cpu_features: Vec<String>,
}

impl Environment {
    /// Create a new environment with required fields.
    #[must_use]
    pub fn new(python_version: Version, torch_version: Version, platform: String) -> Self {
        Self {
            python_version,
            torch_version,
            cuda_version: None,
            cuda_compute: None,
            triton_version: None,
            platform,
            cpu_features: Vec::new(),
        }
    }

    /// Builder pattern: set CUDA version.
    #[must_use]
    pub fn with_cuda_version(mut self, version: Version) -> Self {
        self.cuda_version = Some(version);
        self
    }

    /// Builder pattern: set CUDA compute capability.
    #[must_use]
    pub fn with_cuda_compute(mut self, cc: ComputeCapability) -> Self {
        self.cuda_compute = Some(cc);
        self
    }

    /// Builder pattern: set Triton version.
    #[must_use]
    pub fn with_triton_version(mut self, version: Version) -> Self {
        self.triton_version = Some(version);
        self
    }

    /// Builder pattern: set CPU features.
    #[must_use]
    pub fn with_cpu_features(mut self, features: Vec<String>) -> Self {
        self.cpu_features = features;
        self
    }

    /// Capture the current environment from system/Python.
    ///
    /// This is typically called from Python and passed in, but we provide
    /// a Rust-side default for testing.
    #[must_use]
    pub fn current_default() -> Self {
        let platform = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);

        Self::new(
            Version::new(3, 11, 0), // Placeholder
            Version::new(2, 1, 0),  // Placeholder
            platform,
        )
    }

    /// Check if this environment is compatible with cached environment.
    ///
    /// Returns a detailed compatibility result.
    #[must_use]
    pub fn is_compatible_with(&self, cached: &Environment) -> CompatibilityResult {
        use crate::compat::{
            CompatibilityChecker, CudaComputeRule, PlatformRule, PythonVersionRule,
            TorchVersionRule, TritonVersionRule,
        };

        let checker = CompatibilityChecker::default()
            .with_rule(Box::new(PythonVersionRule))
            .with_rule(Box::new(TorchVersionRule))
            .with_rule(Box::new(CudaComputeRule))
            .with_rule(Box::new(TritonVersionRule))
            .with_rule(Box::new(PlatformRule));

        checker.check(cached, self)
    }
}

/// Result of compatibility check.
#[derive(Debug, Clone)]
pub struct CompatibilityResult {
    /// Overall compatibility status
    pub status: CompatibilityStatus,
    /// Detailed results from each rule
    pub details: Vec<RuleResult>,
}

impl CompatibilityResult {
    /// Check if the environment is compatible (no errors).
    #[must_use]
    pub fn is_compatible(&self) -> bool {
        matches!(
            self.status,
            CompatibilityStatus::Compatible | CompatibilityStatus::Warning
        )
    }

    /// Check if there are any warnings.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        self.details
            .iter()
            .any(|r| matches!(r.status, CompatibilityStatus::Warning))
    }

    /// Get all error messages.
    pub fn errors(&self) -> impl Iterator<Item = &str> {
        self.details.iter().filter_map(|r| {
            if matches!(r.status, CompatibilityStatus::Incompatible) {
                r.message.as_deref()
            } else {
                None
            }
        })
    }

    /// Get all warning messages.
    pub fn warnings(&self) -> impl Iterator<Item = &str> {
        self.details.iter().filter_map(|r| {
            if matches!(r.status, CompatibilityStatus::Warning) {
                r.message.as_deref()
            } else {
                None
            }
        })
    }
}

/// Compatibility status levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityStatus {
    /// Fully compatible, cache can be used
    Compatible,
    /// Compatible with warnings (e.g., newer versions)
    Warning,
    /// Incompatible, cache should not be used
    Incompatible,
}

/// Result from a single compatibility rule.
#[derive(Debug, Clone)]
pub struct RuleResult {
    /// Name of the rule
    pub rule_name: String,
    /// Status from this rule
    pub status: CompatibilityStatus,
    /// Optional message explaining the result
    pub message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parse() {
        let v: Version = "2.1.0".parse().unwrap();
        assert_eq!(v.major, 2);
        assert_eq!(v.minor, 1);
        assert_eq!(v.patch, 0);
        assert!(v.suffix.is_none());
    }

    #[test]
    fn test_version_parse_with_suffix() {
        let v: Version = "2.1.0+cu118".parse().unwrap();
        assert_eq!(v.major, 2);
        assert_eq!(v.minor, 1);
        assert_eq!(v.patch, 0);
        assert_eq!(v.suffix, Some("+cu118".to_string()));
    }

    #[test]
    fn test_version_major_minor_matches() {
        let v1: Version = "2.1.0".parse().unwrap();
        let v2: Version = "2.1.5".parse().unwrap();
        let v3: Version = "2.2.0".parse().unwrap();

        assert!(v1.major_minor_matches(&v2));
        assert!(!v1.major_minor_matches(&v3));
    }

    #[test]
    fn test_version_is_gte() {
        let v1: Version = "2.1.0".parse().unwrap();
        let v2: Version = "2.1.5".parse().unwrap();
        let v3: Version = "2.0.0".parse().unwrap();

        assert!(v2.is_gte(&v1));
        assert!(!v1.is_gte(&v2));
        assert!(v1.is_gte(&v3));
    }

    #[test]
    fn test_compute_capability_parse() {
        let cc: ComputeCapability = "8.0".parse().unwrap();
        assert_eq!(cc.major, 8);
        assert_eq!(cc.minor, 0);
    }

    #[test]
    fn test_compute_capability_is_gte() {
        let cc1: ComputeCapability = "8.0".parse().unwrap();
        let cc2: ComputeCapability = "8.6".parse().unwrap();
        let cc3: ComputeCapability = "7.5".parse().unwrap();

        assert!(cc2.is_gte(&cc1));
        assert!(cc1.is_gte(&cc3));
        assert!(!cc3.is_gte(&cc1));
    }

    #[test]
    fn test_environment_builder() {
        let env = Environment::new(
            Version::new(3, 11, 4),
            Version::new(2, 1, 0),
            "linux-x86_64".to_string(),
        )
        .with_cuda_version(Version::new(11, 8, 0))
        .with_cuda_compute(ComputeCapability::new(8, 0))
        .with_triton_version(Version::new(2, 1, 0));

        assert_eq!(env.python_version.major, 3);
        assert_eq!(env.python_version.minor, 11);
        assert!(env.cuda_version.is_some());
        assert!(env.cuda_compute.is_some());
        assert!(env.triton_version.is_some());
    }

    #[test]
    fn test_environment_serialization() {
        let env = Environment::new(
            Version::new(3, 11, 4),
            Version::new(2, 1, 0),
            "linux-x86_64".to_string(),
        );

        let json = serde_json::to_string(&env).unwrap();
        let parsed: Environment = serde_json::from_str(&json).unwrap();

        assert_eq!(env.python_version.major, parsed.python_version.major);
        assert_eq!(env.torch_version.minor, parsed.torch_version.minor);
    }
}
