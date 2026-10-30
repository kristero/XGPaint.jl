# Changelog

## Unreleased (branch `cluster`): spherical line-of-sight truncation is the default

Breaking change of the default physics. The package version in `Project.toml` is not bumped
yet; bump it (to 0.5.0) before tagging a release from this branch.

### Changed

- All gNFW gas profiles are now spherically truncated by default (`truncation=:spherical`,
  `truncation_radius=4.0`): only gas inside `X R200c` is projected. The line of sight at
  projected radius `x` (R200c units) is the chord `|l| <= sqrt(X^2 - x^2)`, the profile is zero
  for `x >= X`, and `compute_θmax` (and therefore `paint!`) uses `θmax = X θ200c`. This applies
  to `Battaglia16ThermalSZProfile`, `BreakModel`, `BattagliaTauProfile`,
  `BattagliaTauProfilePhysical`, `HaloDMProfile`/`BattagliaFRBProfile`,
  `RSZPerturbativeProfile`, and to `SZPackRSZProfile`/`RKSZpackProfile` through their
  interpolators. Previously every ray was integrated to `1e5 R200c` and only the painted disc
  was cut at `4 θ200c`, so rays at `x < 4` also collected gas from `r > 4 R200c`.
- `profile_grid`/`build_interpolator` of spherical tSZ, tau and DM profiles tabulate the chord
  mean `g = y / (2L)` (positive, smooth across the edge, smoothly continued outside the sphere)
  instead of `y`; the painters and `LogInterpolatorProfile` calls multiply `g` by the exact
  chord `chord_factor(θ, θmax, X) = 2X sqrt(1 - (θ/θmax)^2)`. The numerics are a port of the
  validated HalfDome production engine (`engine/spherical_truncation_profiles.jl`,
  `engine/smooth_exterior.jl`) and reproduce its values to ~1e-15.
- `build_interpolator` cache files store `truncation`, `truncation_radius`, `truncation_zmax`
  and `tabulated_quantity`, and a cache that does not match the model raises an error. Caches
  without these keys are read as legacy cylinder tables. `build_interpolator` also accepts
  `N_z`, `N_logM`, `z_min`, `z_max` and `logM_min` (defaults unchanged).
- `load_precomputed_battaglia` and `load_precomputed_battaglia_tau` wrap their legacy
  infinite line-of-sight tables in `truncation=:cylinder` models.

### Added

- `truncation=:cylinder` (aliases `:cylindrical`, `:projected`, `:infinite`, `:legacy`) on every
  profile constructor: reproduces the previous outputs bit for bit.
- `src/truncation.jl`: `SphereTruncation`, `CylinderTruncation`, `truncation(model)`,
  `truncation_radius(model)`, `tabulates_chord_mean(model)`, `chord_quadrature`,
  `chord_mean`, `smooth_exterior_mean`, `chord_factor`, `los_kernel`, `grid_kernel` (not
  exported). Profile types defined outside XGPaint keep the legacy behaviour.
- `test/test_truncation.jl`.

### Fixed

- `RSZPerturbativeProfile`: `compton_y` called a constructor that no longer existed, its
  callable passed the mass without units, and its CAR `profile_paint!` passed the arguments of
  `profile_paint_generic!` in the wrong order. It now carries a `Battaglia16ThermalSZProfile`
  (`y_model`) with the same truncation keywords.
