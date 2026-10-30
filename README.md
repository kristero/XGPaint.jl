# XGPaint.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://WebSky-CITA.github.io/XGPaint.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://WebSky-CITA.github.io/XGPaint.jl/dev)
[![Build Status](https://github.com/WebSky-CITA/XGPaint.jl/workflows/CI/badge.svg)](https://github.com/WebSky-CITA/XGPaint.jl/actions)
[![codecov](https://codecov.io/gh/WebSky-CITA/XGPaint.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/WebSky-CITA/XGPaint.jl)
<!-- [![Coveralls](https://coveralls.io/repos/github/WebSky-CITA/XGPaint.jl/badge.svg?branch=master)](https://coveralls.io/github/WebSky-CITA/XGPaint.jl?branch=main) -->


`XGPaint.jl` paints maps of extragalactic foregrounds using halo catalogs. Please read the [documentation](https://WebSky-CITA.github.io/XGPaint.jl/dev).

## Line-of-sight truncation (this fork, branch `cluster`)

All gNFW gas profiles (tSZ `Battaglia16ThermalSZProfile` and `BreakModel`, tau/kSZ
`BattagliaTauProfile`, DM/FRB `HaloDMProfile`, and the rSZ/rkSZ wrappers that use them) are
**spherically truncated by default**: only gas inside `r <= X R200c` (`X = 4`) is projected.
At projected radius `x = θ/θ200c` the line of sight is the chord `|l| <= sqrt(X^2 - x^2)`, the
profile is exactly zero for `θ >= X θ200c`, and `paint!` paints each halo out to
`θmax = X θ200c`. Interpolators (`build_interpolator`, `profile_grid`) of these profiles
tabulate the chord mean `y / (2L)`, which is positive and smooth across the edge, and the
painters multiply it by the exact chord `2X sqrt(1 - (θ/θmax)^2)`.

```julia
y_model = Battaglia16ThermalSZProfile(Omega_c=0.2589, Omega_b=0.0486, h=0.6774)  # sphere, X = 4
y_model_3 = Battaglia16ThermalSZProfile(truncation_radius=3.0)                     # sphere, X = 3
y_model_old = Battaglia16ThermalSZProfile(truncation=:cylinder)                    # previous behaviour
```

`truncation=:cylinder` reproduces the previous outputs exactly (line of sight integrated to
`1e5 R200c`, projected disc cut at `4 θ200c`), e.g. to regenerate old maps. The precomputed
WebSky tables (`load_precomputed_battaglia`, `load_precomputed_battaglia_tau`) are legacy
`:cylinder` tables, and `build_interpolator` cache files record their truncation and refuse to
load into a model with a different one. See `CHANGELOG.md`.

