# Changelog

All notable changes to TransitKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-XX-XX

### Added
- **Mandel & Agol transit models** via batman integration
- **Multi-method period detection** (BLS, GLS, PDM) with consensus weighting
- **MCMC parameter estimation** using emcee with full uncertainty quantification
- **Transit Timing Variations (TTV)** detection and measurement
- **Gaussian Process detrending** for systematic removal
- **Injection-recovery testing** for detection efficiency assessment
- **Publication-quality plotting** with AAS/Nature journal styles
- **Validation suite** including odd-even tests, secondary eclipse checks
- **Command-line interface** with rich terminal output
- **GUI application** for interactive analysis
- **NASA Exoplanet Archive** TAP queries for known parameters
- Comprehensive test suite with pytest
- GitHub Actions CI/CD workflows
- ReadTheDocs documentation

### Changed
- Package restructured to `src/` layout
- Improved BLS algorithm with dynamic duration constraints
- Custom PDM implementation (no longer depends on removed astropy function)
- Enhanced error handling throughout

### Fixed
- **BLS duration/period validation error** - durations now dynamically scaled to ensure max_duration < min_period
- **Missing phase_dispersion import** - implemented custom PDM algorithm
- **Missing imports in validation module** - added required core function imports
- Converted all files to Unix line endings

### Deprecated
- `generate_transit_signal()` - use `generate_transit_signal_mandel_agol()`
- `find_transits_box()` - use `find_transits_bls_advanced()`
- `plot_light_curve()` - use `visualization.plot_transit_summary()`

## [1.0.0] - 2023-XX-XX

### Added
- Initial release
- Basic transit signal generation
- Simple BLS detection
- Basic plotting functions
- TESS data loading via lightkurve

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 2.0.0 | 2024 | Major rewrite with advanced features |
| 1.0.0 | 2023 | Initial release |
