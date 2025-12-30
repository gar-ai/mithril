//! Cache compatibility checking.
//!
//! This module provides rules for determining whether a cached artifact
//! is compatible with the current runtime environment.
//!
//! ## Compatibility Rules
//!
//! - **Python Version**: Major.minor must match (3.11.x == 3.11.y)
//! - **PyTorch Version**: Major.minor must match (2.1.x == 2.1.y)
//! - **CUDA Compute**: Current capability >= cached (8.6 >= 8.0)
//! - **Triton Version**: Exact match required (for now)
//! - **Platform**: Must match exactly (linux-x86_64, darwin-arm64)
//!
//! ## Example
//!
//! ```rust
//! use mithril_cache::environment::{Environment, Version};
//! use mithril_cache::compat::CompatibilityChecker;
//!
//! let cached = Environment::new(
//!     Version::new(3, 11, 4),
//!     Version::new(2, 1, 0),
//!     "linux-x86_64".to_string(),
//! );
//!
//! let current = Environment::new(
//!     Version::new(3, 11, 7),  // Same major.minor
//!     Version::new(2, 1, 2),   // Same major.minor
//!     "linux-x86_64".to_string(),
//! );
//!
//! let checker = CompatibilityChecker::default();
//! let result = checker.check(&cached, &current);
//! assert!(result.is_compatible());
//! ```

use crate::environment::{CompatibilityResult, CompatibilityStatus, Environment, RuleResult};

/// A rule for checking compatibility between cached and current environments.
pub trait CompatibilityRule: Send + Sync {
    /// Get the name of this rule.
    fn name(&self) -> &str;

    /// Check compatibility between cached and current environments.
    fn check(&self, cached: &Environment, current: &Environment) -> RuleResult;
}

/// Collection of compatibility rules.
pub struct CompatibilityChecker {
    rules: Vec<Box<dyn CompatibilityRule>>,
}

impl CompatibilityChecker {
    /// Create a new empty checker.
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule to the checker.
    #[must_use]
    pub fn with_rule(mut self, rule: Box<dyn CompatibilityRule>) -> Self {
        self.rules.push(rule);
        self
    }

    /// Check compatibility using all rules.
    pub fn check(&self, cached: &Environment, current: &Environment) -> CompatibilityResult {
        let mut details = Vec::new();
        let mut worst_status = CompatibilityStatus::Compatible;

        for rule in &self.rules {
            let result = rule.check(cached, current);

            // Track worst status
            match result.status {
                CompatibilityStatus::Incompatible => {
                    worst_status = CompatibilityStatus::Incompatible;
                }
                CompatibilityStatus::Warning => {
                    if worst_status != CompatibilityStatus::Incompatible {
                        worst_status = CompatibilityStatus::Warning;
                    }
                }
                CompatibilityStatus::Compatible => {}
            }

            details.push(result);
        }

        CompatibilityResult {
            status: worst_status,
            details,
        }
    }
}

impl Default for CompatibilityChecker {
    fn default() -> Self {
        Self::new()
            .with_rule(Box::new(PythonVersionRule))
            .with_rule(Box::new(TorchVersionRule))
            .with_rule(Box::new(CudaComputeRule))
            .with_rule(Box::new(TritonVersionRule))
            .with_rule(Box::new(PlatformRule))
    }
}

/// Python version compatibility rule.
///
/// Requires major.minor to match (e.g., 3.11.x == 3.11.y).
pub struct PythonVersionRule;

impl CompatibilityRule for PythonVersionRule {
    fn name(&self) -> &str {
        "python_version"
    }

    fn check(&self, cached: &Environment, current: &Environment) -> RuleResult {
        if cached
            .python_version
            .major_minor_matches(&current.python_version)
        {
            RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Compatible,
                message: None,
            }
        } else {
            RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Incompatible,
                message: Some(format!(
                    "Python version mismatch: cached {}.{}, current {}.{}",
                    cached.python_version.major,
                    cached.python_version.minor,
                    current.python_version.major,
                    current.python_version.minor,
                )),
            }
        }
    }
}

/// PyTorch version compatibility rule.
///
/// Requires major.minor to match (e.g., 2.1.x == 2.1.y).
pub struct TorchVersionRule;

impl CompatibilityRule for TorchVersionRule {
    fn name(&self) -> &str {
        "torch_version"
    }

    fn check(&self, cached: &Environment, current: &Environment) -> RuleResult {
        if cached
            .torch_version
            .major_minor_matches(&current.torch_version)
        {
            RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Compatible,
                message: None,
            }
        } else {
            RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Incompatible,
                message: Some(format!(
                    "PyTorch version mismatch: cached {}.{}, current {}.{}",
                    cached.torch_version.major,
                    cached.torch_version.minor,
                    current.torch_version.major,
                    current.torch_version.minor,
                )),
            }
        }
    }
}

/// CUDA compute capability compatibility rule.
///
/// Requires current compute capability >= cached (newer GPU can run older code).
pub struct CudaComputeRule;

impl CompatibilityRule for CudaComputeRule {
    fn name(&self) -> &str {
        "cuda_compute"
    }

    fn check(&self, cached: &Environment, current: &Environment) -> RuleResult {
        match (&cached.cuda_compute, &current.cuda_compute) {
            // Both have CUDA - check compute capability
            (Some(cached_cc), Some(current_cc)) => {
                if current_cc.is_gte(cached_cc) {
                    if current_cc == cached_cc {
                        RuleResult {
                            rule_name: self.name().to_string(),
                            status: CompatibilityStatus::Compatible,
                            message: None,
                        }
                    } else {
                        // Newer compute capability - warn but allow
                        RuleResult {
                            rule_name: self.name().to_string(),
                            status: CompatibilityStatus::Warning,
                            message: Some(format!(
                                "CUDA compute capability differs: cached {}, current {} (forward compatible)",
                                cached_cc, current_cc
                            )),
                        }
                    }
                } else {
                    RuleResult {
                        rule_name: self.name().to_string(),
                        status: CompatibilityStatus::Incompatible,
                        message: Some(format!(
                            "CUDA compute capability too low: cached {}, current {}",
                            cached_cc, current_cc
                        )),
                    }
                }
            }
            // Cached needs CUDA but current doesn't have it
            (Some(cached_cc), None) => RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Incompatible,
                message: Some(format!(
                    "Cache requires CUDA (compute {}), but current environment has no CUDA",
                    cached_cc
                )),
            },
            // Current has CUDA but cached doesn't need it - compatible
            (None, Some(_)) => RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Compatible,
                message: None,
            },
            // Neither has CUDA - compatible
            (None, None) => RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Compatible,
                message: None,
            },
        }
    }
}

/// Triton version compatibility rule.
///
/// Currently requires exact match.
pub struct TritonVersionRule;

impl CompatibilityRule for TritonVersionRule {
    fn name(&self) -> &str {
        "triton_version"
    }

    fn check(&self, cached: &Environment, current: &Environment) -> RuleResult {
        match (&cached.triton_version, &current.triton_version) {
            // Both have Triton - require exact major.minor match
            (Some(cached_v), Some(current_v)) => {
                if cached_v.major_minor_matches(current_v) {
                    RuleResult {
                        rule_name: self.name().to_string(),
                        status: CompatibilityStatus::Compatible,
                        message: None,
                    }
                } else {
                    RuleResult {
                        rule_name: self.name().to_string(),
                        status: CompatibilityStatus::Incompatible,
                        message: Some(format!(
                            "Triton version mismatch: cached {}.{}, current {}.{}",
                            cached_v.major, cached_v.minor, current_v.major, current_v.minor,
                        )),
                    }
                }
            }
            // Cached needs Triton but current doesn't have it
            (Some(cached_v), None) => RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Incompatible,
                message: Some(format!(
                    "Cache requires Triton {}.{}, but Triton not available",
                    cached_v.major, cached_v.minor
                )),
            },
            // Current has Triton but cached doesn't need it - compatible
            (None, Some(_)) => RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Compatible,
                message: None,
            },
            // Neither has Triton - compatible
            (None, None) => RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Compatible,
                message: None,
            },
        }
    }
}

/// Platform compatibility rule.
///
/// Requires exact platform match.
pub struct PlatformRule;

impl CompatibilityRule for PlatformRule {
    fn name(&self) -> &str {
        "platform"
    }

    fn check(&self, cached: &Environment, current: &Environment) -> RuleResult {
        if cached.platform == current.platform {
            RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Compatible,
                message: None,
            }
        } else {
            RuleResult {
                rule_name: self.name().to_string(),
                status: CompatibilityStatus::Incompatible,
                message: Some(format!(
                    "Platform mismatch: cached '{}', current '{}'",
                    cached.platform, current.platform
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::{ComputeCapability, Version};

    fn make_env(
        py_major: u32,
        py_minor: u32,
        torch_major: u32,
        torch_minor: u32,
        platform: &str,
    ) -> Environment {
        Environment::new(
            Version::new(py_major, py_minor, 0),
            Version::new(torch_major, torch_minor, 0),
            platform.to_string(),
        )
    }

    #[test]
    fn test_python_version_compatible() {
        let cached = make_env(3, 11, 2, 1, "linux-x86_64");
        let current = make_env(3, 11, 2, 1, "linux-x86_64");

        let rule = PythonVersionRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Compatible));
    }

    #[test]
    fn test_python_version_patch_difference() {
        let cached = Environment::new(
            Version::new(3, 11, 4),
            Version::new(2, 1, 0),
            "linux-x86_64".to_string(),
        );
        let current = Environment::new(
            Version::new(3, 11, 7),
            Version::new(2, 1, 0),
            "linux-x86_64".to_string(),
        );

        let rule = PythonVersionRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Compatible));
    }

    #[test]
    fn test_python_version_incompatible() {
        let cached = make_env(3, 10, 2, 1, "linux-x86_64");
        let current = make_env(3, 11, 2, 1, "linux-x86_64");

        let rule = PythonVersionRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Incompatible));
        assert!(result.message.is_some());
    }

    #[test]
    fn test_torch_version_compatible() {
        let cached = make_env(3, 11, 2, 1, "linux-x86_64");
        let current = make_env(3, 11, 2, 1, "linux-x86_64");

        let rule = TorchVersionRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Compatible));
    }

    #[test]
    fn test_torch_version_incompatible() {
        let cached = make_env(3, 11, 2, 0, "linux-x86_64");
        let current = make_env(3, 11, 2, 1, "linux-x86_64");

        let rule = TorchVersionRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Incompatible));
    }

    #[test]
    fn test_cuda_compute_compatible_same() {
        let cached =
            make_env(3, 11, 2, 1, "linux-x86_64").with_cuda_compute(ComputeCapability::new(8, 0));
        let current =
            make_env(3, 11, 2, 1, "linux-x86_64").with_cuda_compute(ComputeCapability::new(8, 0));

        let rule = CudaComputeRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Compatible));
    }

    #[test]
    fn test_cuda_compute_compatible_newer() {
        let cached =
            make_env(3, 11, 2, 1, "linux-x86_64").with_cuda_compute(ComputeCapability::new(8, 0));
        let current =
            make_env(3, 11, 2, 1, "linux-x86_64").with_cuda_compute(ComputeCapability::new(8, 6));

        let rule = CudaComputeRule;
        let result = rule.check(&cached, &current);

        // Newer compute capability produces a warning but is compatible
        assert!(matches!(result.status, CompatibilityStatus::Warning));
    }

    #[test]
    fn test_cuda_compute_incompatible_older() {
        let cached =
            make_env(3, 11, 2, 1, "linux-x86_64").with_cuda_compute(ComputeCapability::new(8, 6));
        let current =
            make_env(3, 11, 2, 1, "linux-x86_64").with_cuda_compute(ComputeCapability::new(8, 0));

        let rule = CudaComputeRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Incompatible));
    }

    #[test]
    fn test_cuda_compute_missing_current() {
        let cached =
            make_env(3, 11, 2, 1, "linux-x86_64").with_cuda_compute(ComputeCapability::new(8, 0));
        let current = make_env(3, 11, 2, 1, "linux-x86_64");

        let rule = CudaComputeRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Incompatible));
    }

    #[test]
    fn test_platform_compatible() {
        let cached = make_env(3, 11, 2, 1, "linux-x86_64");
        let current = make_env(3, 11, 2, 1, "linux-x86_64");

        let rule = PlatformRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Compatible));
    }

    #[test]
    fn test_platform_incompatible() {
        let cached = make_env(3, 11, 2, 1, "linux-x86_64");
        let current = make_env(3, 11, 2, 1, "darwin-arm64");

        let rule = PlatformRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Incompatible));
    }

    #[test]
    fn test_checker_all_compatible() {
        let cached = make_env(3, 11, 2, 1, "linux-x86_64");
        let current = make_env(3, 11, 2, 1, "linux-x86_64");

        let checker = CompatibilityChecker::default();
        let result = checker.check(&cached, &current);

        assert!(result.is_compatible());
        assert!(!result.has_warnings());
    }

    #[test]
    fn test_checker_with_warning() {
        let cached =
            make_env(3, 11, 2, 1, "linux-x86_64").with_cuda_compute(ComputeCapability::new(8, 0));
        let current =
            make_env(3, 11, 2, 1, "linux-x86_64").with_cuda_compute(ComputeCapability::new(8, 6));

        let checker = CompatibilityChecker::default();
        let result = checker.check(&cached, &current);

        assert!(result.is_compatible());
        assert!(result.has_warnings());
    }

    #[test]
    fn test_checker_incompatible() {
        let cached = make_env(3, 11, 2, 1, "linux-x86_64");
        let current = make_env(3, 10, 2, 1, "linux-x86_64");

        let checker = CompatibilityChecker::default();
        let result = checker.check(&cached, &current);

        assert!(!result.is_compatible());
        assert!(result.errors().next().is_some());
    }

    #[test]
    fn test_checker_multiple_failures() {
        let cached = make_env(3, 11, 2, 1, "linux-x86_64");
        let current = make_env(3, 10, 2, 0, "darwin-arm64");

        let checker = CompatibilityChecker::default();
        let result = checker.check(&cached, &current);

        assert!(!result.is_compatible());
        let errors: Vec<_> = result.errors().collect();
        assert!(errors.len() >= 2); // At least Python and platform failures
    }

    #[test]
    fn test_triton_version_compatible() {
        let cached =
            make_env(3, 11, 2, 1, "linux-x86_64").with_triton_version(Version::new(2, 1, 0));
        let current =
            make_env(3, 11, 2, 1, "linux-x86_64").with_triton_version(Version::new(2, 1, 3));

        let rule = TritonVersionRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Compatible));
    }

    #[test]
    fn test_triton_version_incompatible() {
        let cached =
            make_env(3, 11, 2, 1, "linux-x86_64").with_triton_version(Version::new(2, 0, 0));
        let current =
            make_env(3, 11, 2, 1, "linux-x86_64").with_triton_version(Version::new(2, 1, 0));

        let rule = TritonVersionRule;
        let result = rule.check(&cached, &current);

        assert!(matches!(result.status, CompatibilityStatus::Incompatible));
    }
}
