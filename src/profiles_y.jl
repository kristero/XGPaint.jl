"""Thermal Sunyaev-Zel'dovich profiles, based on Battaglia et al. 2016. We also include a
break model variant.

    Battaglia16ThermalSZProfile(; Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
        P0_amp=18.1, ..., gamma_alpha_z=0.0, truncation=:spherical, truncation_radius=4.0)

Compton-y profile of the Battaglia et al. (2012/2016) gNFW pressure fit, `model(θ, M200c, z)`
with θ in radians and M200c in Msun.

Line-of-sight truncation (see `SphereTruncation` / `CylinderTruncation`):
- `truncation=:spherical` (default): only gas inside the sphere of radius
  `truncation_radius * R200c` is projected; y is zero for θ >= X θ200c, which is also the
  painting radius returned by `compute_θmax`.
- `truncation=:cylinder`: the previous behaviour (LOS integrated to 1e5 R200c, projected disc
  cut at `truncation_radius * θ200c`), which reproduces earlier XGPaint outputs.
"""
struct Battaglia16ThermalSZProfile{T,C,R<:AbstractTruncation} <: AbstractGNFW{T}
    f_b::T  # Omega_b / Omega_c = 0.0486 / 0.2589
    cosmo::C
    P0::PowerLawParam{T}
    x_c::PowerLawParam{T}
    alpha::PowerLawParam{T}
    beta::PowerLawParam{T}
    gamma::PowerLawParam{T}
    truncation::R
end

"""
    BreakModel(; ..., alpha_break=1.5, M_break=2e14, truncation=:spherical, truncation_radius=4.0)

Battaglia16 tSZ profile with a power-law suppression `(M/M_break)^alpha_break` below
`M_break`. The `truncation` keywords are the same as for `Battaglia16ThermalSZProfile`.
"""
struct BreakModel{T,C,R<:AbstractTruncation} <: AbstractGNFW{T}
    f_b::T
    cosmo::C
    P0::PowerLawParam{T}
    x_c::PowerLawParam{T}
    alpha::PowerLawParam{T}
    beta::PowerLawParam{T}
    gamma::PowerLawParam{T}
    alpha_break::T
    M_break::T
    truncation::R
end

# positional constructors with the pre-truncation field list get the default truncation
Battaglia16ThermalSZProfile(f_b, cosmo, P0, x_c, alpha, beta, gamma) =
    Battaglia16ThermalSZProfile(f_b, cosmo, P0, x_c, alpha, beta, gamma, default_truncation())
BreakModel(f_b, cosmo, P0, x_c, alpha, beta, gamma, alpha_break, M_break) =
    BreakModel(f_b, cosmo, P0, x_c, alpha, beta, gamma, alpha_break, M_break, default_truncation())

function Battaglia16ThermalSZProfile(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774,
        P0_amp::T=18.1, P0_alpha_m::T=0.154, P0_alpha_z::T=-0.758,
        x_c_amp::T=0.497, x_c_alpha_m::T=-0.00865, x_c_alpha_z::T=0.731,
        alpha_amp::T=1.0, alpha_alpha_m::T=0.0, alpha_alpha_z::T=0.0,
        beta_amp::T=4.35, beta_alpha_m::T=0.0393, beta_alpha_z::T=0.415,
        gamma_amp::T=-0.3, gamma_alpha_m::T=0.0, gamma_alpha_z::T=0.0,
        truncation=:spherical, truncation_radius=DEFAULT_TRUNCATION_RADIUS) where {T <: Real}
    OmegaM=Omega_b+Omega_c
    f_b = Omega_b / OmegaM
    cosmo = get_cosmology(T, h=h, OmegaM=OmegaM)
    P0 = PowerLawParam(T(P0_amp), T(P0_alpha_m), T(P0_alpha_z))
    x_c = PowerLawParam(T(x_c_amp), T(x_c_alpha_m), T(x_c_alpha_z))
    alpha = PowerLawParam(T(alpha_amp), T(alpha_alpha_m), T(alpha_alpha_z))
    beta = PowerLawParam(T(beta_amp), T(beta_alpha_m), T(beta_alpha_z))
    gamma = PowerLawParam(T(gamma_amp), T(gamma_alpha_m), T(gamma_alpha_z))
    tr = make_truncation(truncation, truncation_radius)
    return Battaglia16ThermalSZProfile(f_b, cosmo, P0, x_c, alpha, beta, gamma, tr)
end

function BreakModel(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774,
        alpha_break::T=1.5, M_break::T=2.0*10^14,
        P0_amp::T=18.1, P0_alpha_m::T=0.154, P0_alpha_z::T=-0.758,
        x_c_amp::T=0.497, x_c_alpha_m::T=-0.00865, x_c_alpha_z::T=0.731,
        alpha_amp::T=1.0, alpha_alpha_m::T=0.0, alpha_alpha_z::T=0.0,
        beta_amp::T=4.35, beta_alpha_m::T=0.0393, beta_alpha_z::T=0.415,
        gamma_amp::T=-0.3, gamma_alpha_m::T=0.0, gamma_alpha_z::T=0.0,
        truncation=:spherical, truncation_radius=DEFAULT_TRUNCATION_RADIUS) where {T <: Real}
    #alpha_break = 1.486 from Shivam P paper by Nate's sleuthing
    OmegaM=Omega_b+Omega_c
    f_b = Omega_b / OmegaM
    cosmo = get_cosmology(T, h=h, OmegaM=OmegaM)
    P0 = PowerLawParam(T(P0_amp), T(P0_alpha_m), T(P0_alpha_z))
    x_c = PowerLawParam(T(x_c_amp), T(x_c_alpha_m), T(x_c_alpha_z))
    alpha = PowerLawParam(T(alpha_amp), T(alpha_alpha_m), T(alpha_alpha_z))
    beta = PowerLawParam(T(beta_amp), T(beta_alpha_m), T(beta_alpha_z))
    gamma = PowerLawParam(T(gamma_amp), T(gamma_alpha_m), T(gamma_alpha_z))
    tr = make_truncation(truncation, truncation_radius)
    return BreakModel(f_b, cosmo, P0, x_c, alpha, beta, gamma, alpha_break, M_break, tr)
end

truncation(model::Union{Battaglia16ThermalSZProfile,BreakModel}) = model.truncation
tabulates_chord_mean(model::Union{Battaglia16ThermalSZProfile,BreakModel}) =
    model.truncation isa SphereTruncation

function generalized_nfw(x, xc, α, β, γ)
    x̄ = x / xc
    return x̄^γ * (1 + x̄^α)^(-(β + γ) / α)    # correction to battaglia 2016 tau.
end

function _generalized_scaled_nfw(x̄, α, β, γ)
    return x̄^γ * (1 + x̄^α)^(-(β + γ) / α)
end


function get_params(model::Battaglia16ThermalSZProfile{T}, M_200, z) where T
	z₁ = z + 1
	m = M_200 / (1e14M_sun)
	P₀ = powerlaw_value(model.P0, m, z₁)
	xc = powerlaw_value(model.x_c, m, z₁)
	α = powerlaw_value(model.alpha, m, z₁)
	β_raw = powerlaw_value(model.beta, m, z₁)
    γ = powerlaw_value(model.gamma, m, z₁)
    # Battaglia parameterizes the outer factor as (1 + x^α)^(-β_raw), while
    # generalized_nfw expects the asymptotic outer slope directly. Matching the
    # large-radius behavior x^(γ - αβ_raw) = x^(-β) gives β = αβ_raw - γ.
    β = α * β_raw - γ
    return (xc=T(xc), α=T(α), β=T(β), γ=T(γ), P₀=T(P₀))
end

function get_params(model::BreakModel{T}, M_200, z) where T
    z₁ = z + 1
    m = M_200 / (1e14M_sun)
    P₀ = powerlaw_value(model.P0, m, z₁)
    xc = powerlaw_value(model.x_c, m, z₁)
    α = powerlaw_value(model.alpha, m, z₁)
    β_raw = powerlaw_value(model.beta, m, z₁)
    γ = powerlaw_value(model.gamma, m, z₁)
    β = α * β_raw - γ
    return (xc=T(xc), α=T(α), β=T(β), γ=T(γ), P₀=T(P₀))
end

"""
    _nfw_profile_los_quadrature(x, xc, α, β, γ; zmax=1e5, rtol=1e-12, order=9)

Legacy (effectively infinite) line-of-sight integral `2 ∫_0^zmax p(sqrt(l^2 + x^2)) dl` of
the gNFW shape, in units of R200c. Used by `CylinderTruncation`; spherical profiles use
`chord_quadrature` instead (see `los_kernel`).
"""
function _nfw_profile_los_quadrature(x, xc, α, β, γ; zmax=1e5, rtol=1e-12, order=9)
    x² = x^2
    scale = 1e9
    integral, err = quadgk(y -> scale * generalized_nfw(√(y^2 + x²), xc, α, β, γ),
                      0.0, zmax, rtol=rtol, order=order)
    return 2integral / scale
end

function dimensionless_P_profile_los(model::Battaglia16ThermalSZProfile{T}, r, M_200, z) where T
    par = get_params(model, M_200, z)
    R_200 = R_Δ(model, M_200, z, 200)
    x = r / angular_size(model, R_200, z)
    return par.P₀ * los_kernel(model.truncation, x, par.xc, par.α, par.β, par.γ)
end

function dimensionless_P_profile_los(model::BreakModel{T}, r, M_200, z) where T
    par = get_params(model, M_200, z)
    R_200 = R_Δ(model, M_200, z, 200)
    x = r / angular_size(model, R_200, z)
    if M_200 < model.M_break * M_sun
        return (par.P₀ * (M_200/(model.M_break*M_sun))^model.alpha_break *
            los_kernel(model.truncation, x, par.xc, par.α, par.β, par.γ))
    else
        return par.P₀ * los_kernel(model.truncation, x, par.xc, par.α, par.β, par.γ)
    end
end

"""Line-of-sight integrated electron pressure"""
P_e_los(model, r, M_200c, z) = 0.5176 * P_th_los(model, r, M_200c, z)

"""Line-of-sight integrated thermal pressure"""
P_th_los(model, r, M_200c, z) = constants.G * M_200c * 200 * ρ_crit(model, z) *
    model.f_b / 2 * dimensionless_P_profile_los(model, r, M_200c, z)

"""
    compton_y(model, r, M_200c, z)

Calculate the Compton y parameter for a given model at a given radius, mass, and redshift.
Mass needs to have units! The line of sight follows `truncation(model)` (spherical by
default: zero for r >= X θ200c).
"""
function compton_y(model::Battaglia16ThermalSZProfile, r, M_200c, z)
    return P_e_los(model, r, M_200c, z) * P_e_factor + 0   # +0 to strip units
end
function compton_y(model::BreakModel, r, M_200c, z)
    return P_e_los(model, r, M_200c, z) * P_e_factor + 0   # +0 to strip units
end

# Grid fast path used by `profile_grid`: one parameter evaluation per (M, z) slice. The
# tabulated kernel is `grid_kernel(truncation, x, ...)`: the LOS column for the cylinder,
# the chord mean g = y / (2L) for the sphere (see `tabulates_chord_mean`).
function prepare_profile_slice(model::Battaglia16ThermalSZProfile{T}, mass, redshift) where T
    mass_with_units = mass * M_sun
    par = get_params(model, mass_with_units, redshift)
    alpha = getproperty(par, Symbol("α"))
    beta = getproperty(par, Symbol("β"))
    gamma = getproperty(par, Symbol("γ"))
    p0 = getproperty(par, Symbol("P₀"))
    r200 = var"R_Δ"(model, mass_with_units, redshift, 200)
    theta_scale = T(angular_size(model, r200, redshift))
    amplitude = T(
        (
            0.5176 *
            constants.G *
            mass_with_units *
            200 *
            var"ρ_crit"(model, redshift) *
            model.f_b /
            2 *
            P_e_factor *
            p0
        ) + 0
    )
    return (; theta_scale, xc=par.xc, alpha, beta, gamma, amplitude, truncation=model.truncation)
end

@inline function evaluate_profile_slice(::Battaglia16ThermalSZProfile, prepared, theta, mass, redshift)
    x = theta / prepared.theta_scale
    return prepared.amplitude * grid_kernel(
        prepared.truncation,
        x,
        prepared.xc,
        prepared.alpha,
        prepared.beta,
        prepared.gamma,
    )
end

function prepare_profile_slice(model::BreakModel{T}, mass, redshift) where T
    mass_with_units = mass * M_sun
    par = get_params(model, mass_with_units, redshift)
    alpha = getproperty(par, Symbol("α"))
    beta = getproperty(par, Symbol("β"))
    gamma = getproperty(par, Symbol("γ"))
    p0 = getproperty(par, Symbol("P₀"))
    r200 = var"R_Δ"(model, mass_with_units, redshift, 200)
    theta_scale = T(angular_size(model, r200, redshift))
    break_factor = mass_with_units < model.M_break * M_sun ?
        (mass_with_units / (model.M_break * M_sun))^model.alpha_break :
        one(T)
    amplitude = T(
        (
            0.5176 *
            constants.G *
            mass_with_units *
            200 *
            var"ρ_crit"(model, redshift) *
            model.f_b /
            2 *
            P_e_factor *
            p0 *
            break_factor
        ) + 0
    )
    return (; theta_scale, xc=par.xc, alpha, beta, gamma, amplitude, truncation=model.truncation)
end

@inline function evaluate_profile_slice(::BreakModel, prepared, theta, mass, redshift)
    x = theta / prepared.theta_scale
    return prepared.amplitude * grid_kernel(
        prepared.truncation,
        x,
        prepared.xc,
        prepared.alpha,
        prepared.beta,
        prepared.gamma,
    )
end


# direct evaluation of a model at a given radius, mass, and redshift.
# mass is in Msun; Unitful is NOT used in these evaluation functions
(model::Battaglia16ThermalSZProfile)(r, M_200c, z) = compton_y(model, r, M_200c * M_sun, z)
(model::BreakModel)(r, M_200c, z) = compton_y(model, r, M_200c * M_sun, z)
