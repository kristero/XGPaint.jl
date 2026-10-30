# Line-of-sight truncation of projected gNFW profiles.
#
# Every gNFW profile in this package (tSZ `Battaglia16ThermalSZProfile` and `BreakModel`,
# tau/kSZ `BattagliaTauProfile`, DM/FRB `HaloDMProfile`, and the SZ wrappers built on them)
# is a projection of a 3-D shape p(r) with r in units of R200c. The truncation decides
# which gas enters that projection:
#
# - SphereTruncation (the default): only gas inside the sphere r <= X R200c is projected.
#   At projected radius x the line of sight is the chord |l| <= L(x) = sqrt(X^2 - x^2), the
#   column is exactly zero for x >= X, and the painting radius is θmax = X θ200c. X = 4.
# - CylinderTruncation (the previous behaviour): every ray is integrated over
#   |l| <= 1e5 R200c (effectively infinite), and only the painted disc is cut at
#   θmax = X θ200c (X = 4). Select it with `truncation=:cylinder` to regenerate old maps.
#
# Interpolators of spherical profiles tabulate the chord mean g(x) = y(x) / (2 L(x)), which
# is positive and smooth, continuous at the edge (g(X) = p(X)), and is continued smoothly
# outside the sphere (never painted). Painters multiply the interpolated g by the exact
# chord 2 L = 2 X sqrt(1 - (θ/θmax)^2) (see `chord_factor`). This follows the production
# HalfDome implementation (engine/spherical_truncation_profiles.jl, engine/smooth_exterior.jl).

"""
    AbstractTruncation

Supertype of the line-of-sight truncations, see [`SphereTruncation`](@ref) and
[`CylinderTruncation`](@ref).
"""
abstract type AbstractTruncation end

"""
    SphereTruncation(X=4.0)

Count only gas inside the sphere of radius `X R200c`. The LOS integral at projected radius
`x` (R200c units) runs over `|l| <= sqrt(X^2 - x^2)` and is zero for `x >= X`; the painted
disc is `θ < θmax = X θ200c`. This is the default truncation of all gNFW profiles.
"""
struct SphereTruncation <: AbstractTruncation
    X::Float64
    function SphereTruncation(X::Real=DEFAULT_TRUNCATION_RADIUS)
        (isfinite(X) && X > 0) || throw(ArgumentError(
            "sphere radius X must be finite and positive (units of R200c), got $(X)"))
        return new(Float64(X))
    end
end

"""
    CylinderTruncation(X=4.0, zmax=1e5)

Legacy (XGPaint <= 0.4, WebSky) treatment: the LOS integral runs over `|l| <= zmax R200c`
(1e5 R200c, effectively infinite) at every projected radius, and only the painted disc is
cut at `θmax = X θ200c`. Select it with the keyword `truncation=:cylinder` (aliases
`:cylindrical`, `:projected`, `:infinite`, `:legacy`) to reproduce previous outputs.
"""
struct CylinderTruncation <: AbstractTruncation
    X::Float64
    zmax::Float64
    function CylinderTruncation(X::Real=DEFAULT_TRUNCATION_RADIUS, zmax::Real=LEGACY_LOS_ZMAX)
        (isfinite(X) && X > 0) || throw(ArgumentError(
            "projected cut X must be finite and positive (units of R200c), got $(X)"))
        (zmax > 0) || throw(ArgumentError("zmax must be positive, got $(zmax)"))
        return new(Float64(X), Float64(zmax))
    end
end

"Default truncation radius X in units of R200c (sphere radius, and painting radius θmax = X θ200c)."
const DEFAULT_TRUNCATION_RADIUS = 4.0
"LOS half-length (units of R200c) of the legacy, effectively infinite, line of sight."
const LEGACY_LOS_ZMAX = 1e5

"""
    make_truncation(spec, X=4.0)

Build an [`AbstractTruncation`](@ref) from the `truncation` keyword of the profile
constructors. `spec` is `:spherical` (alias `:sphere`) for [`SphereTruncation`](@ref)`(X)`,
`:cylinder` (aliases `:cylindrical`, `:projected`, `:infinite`, `:legacy`) for
[`CylinderTruncation`](@ref)`(X)`, or an `AbstractTruncation`, which is returned unchanged
(its own radius wins over `X`).
"""
make_truncation(t::AbstractTruncation, X=DEFAULT_TRUNCATION_RADIUS) = t
make_truncation(spec::AbstractString, X=DEFAULT_TRUNCATION_RADIUS) = make_truncation(Symbol(spec), X)
function make_truncation(spec::Symbol, X=DEFAULT_TRUNCATION_RADIUS)
    spec in (:spherical, :sphere) && return SphereTruncation(X)
    spec in (:cylinder, :cylindrical, :projected, :infinite, :legacy) && return CylinderTruncation(X)
    throw(ArgumentError("unknown truncation $(repr(spec)); use :spherical (default) or :cylinder"))
end

"Truncation used when a profile is constructed without the `truncation` keyword."
default_truncation() = SphereTruncation(DEFAULT_TRUNCATION_RADIUS)

"""
    truncation(model)

The [`AbstractTruncation`](@ref) of a profile. Profiles that do not carry one (user-defined
profile types, test models, wrappers defined outside XGPaint) keep the legacy behaviour,
`CylinderTruncation(4.0, 1e5)`, i.e. the painters treat them exactly as before.
"""
truncation(::Any) = CylinderTruncation(DEFAULT_TRUNCATION_RADIUS, LEGACY_LOS_ZMAX)

"""
    truncation_radius(model)

`X` of `truncation(model)`: the painting radius θmax = X θ200c used by `compute_θmax`
(and, for spherical profiles, the sphere radius in units of R200c).
"""
truncation_radius(model) = truncation(model).X

is_spherical(model) = truncation(model) isa SphereTruncation

"""
    tabulates_chord_mean(model)

`true` when `profile_grid(model)` (and therefore `build_interpolator(model)`) tabulates the
chord mean `g = y / (2L)` instead of the projected profile `y`. This holds for the
spherically truncated tSZ, tau and DM profiles of this package; the painters then multiply
the interpolated `g` by [`chord_factor`](@ref). All other models tabulate `y` itself.
"""
tabulates_chord_mean(::Any) = false

_tr_float(x::Real) = Float64(x)
_tr_float(x::Unitful.Quantity) = Float64(uconvert(Unitful.NoUnits, x))

# ---------------------------------------------------------------------------------------
# Chord quadrature, ported from the production HalfDome engine
# (engine/spherical_truncation_profiles.jl l.33-84, engine/smooth_exterior.jl l.8-30).
# All arithmetic is Float64; `β` is XGPaint's asymptotic outer slope (for tSZ get_params
# already converts β_raw to α β_raw - γ).
# ---------------------------------------------------------------------------------------

@inline log1pexp_stable(v) = max(v, 0.0) + log1p(exp(-abs(v)))

"""Log of r times the gNFW shape, i.e. the LOS integrand including the Jacobian dl = r du."""
@inline function log_radial_integrand(logr, xc, α, β, γ)
    exponent = (β + γ) / α
    return (1 + γ) * logr - γ * log(xc) -
           exponent * log1pexp_stable(α * (logr - log(xc)))
end

"""Peak log amplitude of the integrand over [loglo, loghi]; only a numerical scale."""
function log_integrand_peak(loglo, loghi, xc, α, β, γ)
    peak = log_radial_integrand(loghi, xc, α, β, γ)
    isfinite(loglo) && (peak = max(peak, log_radial_integrand(loglo, xc, α, β, γ)))
    if 1 + γ > 0 && β > 1
        stationary = log(xc) + log((1 + γ) / (β - 1)) / α
        at = clamp(stationary, loglo, loghi)
        peak = max(peak, log_radial_integrand(at, xc, α, β, γ))
    end
    return peak
end

"""QuadGK with a hard accuracy check: the value must be finite, positive and converged."""
function checked_quadrature(f, lo, hi; rtol=1.0e-10, maxevals=4096)
    value, error = quadgk(f, lo, hi; rtol=rtol, atol=0.0, order=9, maxevals=maxevals)
    isfinite(value) && value > 0 && isfinite(error) && error <= rtol * value ||
        throw(ErrorException("Finite-chord quadrature failed: value=$value error=$error rtol=$rtol maxevals=$maxevals"))
    return value
end

"""
    chord_quadrature(x, xc, α, β, γ, L; rtol=1e-10, maxevals=4096)

Dimensionless chord column `∫_{-L}^{L} p(sqrt(x^2 + l^2)) dl` of the gNFW shape
`p(r) = (r/xc)^γ (1 + (r/xc)^α)^(-(β+γ)/α)` (all lengths in R200c). For `x > 0` the
substitution `l = x sinh(u)`, `r = x cosh(u)` is used; at `x = 0` the integral is done in
`log r`, which handles the integrable central cusp (requires `γ > -1`). The integrand is
normalised by its peak and the result is checked by [`checked_quadrature`](@ref).
Returns 0 for `L <= 0`.
"""
function chord_quadrature(x, xc, α, β, γ, L; rtol=1.0e-10, maxevals=4096)
    L <= 0 && return 0.0
    x >= 0 && xc > 0 && α > 0 || error("invalid gNFW geometry/scale")
    γ > -1 || x > 0 || error("central LOS is divergent for gamma <= -1")
    loghi = log(hypot(x, L))
    loglo = x == 0 ? -Inf : log(x)
    peak = log_integrand_peak(loglo, loghi, xc, α, β, γ)
    if x == 0
        central = logr -> exp(log_radial_integrand(logr, xc, α, β, γ) - peak)
        I = checked_quadrature(central, -Inf, loghi; rtol=rtol, maxevals=maxevals)
    else
        offset = u -> exp(log_radial_integrand(log(x) + log(cosh(u)), xc, α, β, γ) - peak)
        I = checked_quadrature(offset, 0.0, asinh(L / x); rtol=rtol, maxevals=maxevals)
    end
    return exp(log(2I) + peak)
end

# 16-point Gauss-Legendre rule on [0, 1] (production reads the same rule from legendre16.csv)
const _GL16_NODES, _GL16_WEIGHTS = QuadGK.gauss(16, 0.0, 1.0)

"""
    smooth_exterior_mean(x, xc, α, β, γ, X)

`h(x) = ∫_0^1 p(sqrt(x^2 (1 - u^2) + X^2 u^2)) du`. Inside the sphere this is the exact chord
mean; for `x >= X` it is a positive analytic continuation used only to keep the cached
`log g` smooth across the edge (`h(X) = p(X)`). Never painted: the chord factor is 0 there.
"""
function smooth_exterior_mean(x, xc, α, β, γ, X)
    logs = ntuple(16) do i
        u = _GL16_NODES[i]
        radius = sqrt(x * x * (1 - u * u) + X * X * u * u)
        logq = log(radius / xc)
        γ * logq - ((β + γ) / α) * log1pexp_stable(α * logq)
    end
    peak = maximum(logs)
    total = sum(_GL16_WEIGHTS[i] * exp(logs[i] - peak) for i in 1:16)
    return exp(peak + log(total))
end

"""
    chord_mean(x, xc, α, β, γ, X)

Cached quantity of spherically truncated profiles: `chord_quadrature(x, ..., L) / (2L)`
with `L = sqrt(X^2 - x^2)` for `x < X`, and [`smooth_exterior_mean`](@ref) for `x >= X`.
Positive, finite and continuous for all `x > 0`.
"""
function chord_mean(x, xc, α, β, γ, X)
    if x < X
        L = sqrt((X - x) * (X + x))
        return chord_quadrature(x, xc, α, β, γ, L) / (2L)
    end
    return smooth_exterior_mean(x, xc, α, β, γ, X)
end

"""
    chord_factor(θ, θmax, X)

Exact chord length `2 L = 2 X sqrt(1 - (θ/θmax)^2)` (units of R200c) for `θmax = X θ200c`,
and 0 for `θ >= θmax`. Multiplying the cached chord mean `g` by it gives the column `y`.
"""
@inline function chord_factor(θ, θmax, X)
    ratio = θ / θmax
    ratio >= 1 && return 0.0
    return 2X * sqrt(1 - ratio^2)
end

# ---------------------------------------------------------------------------------------
# Kernels used by the profile code
# ---------------------------------------------------------------------------------------

"""
    los_kernel(truncation, x, xc, α, β, γ)

Dimensionless LOS column of the gNFW shape at projected radius `x` (units of R200c):
the legacy `_nfw_profile_los_quadrature(x, ...; zmax)` for [`CylinderTruncation`](@ref)
(bit-for-bit the previous numbers), the finite chord for [`SphereTruncation`](@ref)
(exactly 0 for `x >= X`).
"""
@inline los_kernel(t::CylinderTruncation, x, xc, α, β, γ) =
    _nfw_profile_los_quadrature(x, xc, α, β, γ; zmax=t.zmax)

function los_kernel(t::SphereTruncation, x, xc, α, β, γ)
    xf = _tr_float(x)
    X = t.X
    xf >= X && return 0.0
    return chord_quadrature(xf, Float64(xc), Float64(α), Float64(β), Float64(γ),
                            sqrt((X - xf) * (X + xf)))
end

"""
    grid_kernel(truncation, x, xc, α, β, γ)

Kernel tabulated by `profile_grid`: the LOS column for [`CylinderTruncation`](@ref), the
chord mean `g` ([`chord_mean`](@ref)) for [`SphereTruncation`](@ref).
"""
@inline grid_kernel(t::CylinderTruncation, x, xc, α, β, γ) = los_kernel(t, x, xc, α, β, γ)
@inline grid_kernel(t::SphereTruncation, x, xc, α, β, γ) = chord_mean(
    _tr_float(x), Float64(xc), Float64(α), Float64(β), Float64(γ), t.X)
