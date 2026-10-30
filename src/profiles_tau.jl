
# can be used with either angular (default) or physical units
abstract type AbstractBattagliaTauProfile{T} <: AbstractGNFW{T} end

"""
    BattagliaTauProfile(; Omega_c=0.2589, Omega_b=0.0486, h=0.6774, P0_amp=4.0e3, ...,
        truncation=:spherical, truncation_radius=4.0)

Optical depth (kSZ) profile from the Battaglia (2016) gas-density fit, `model(θ, M200c, z)`
with θ in radians and M200c in Msun. The projected electron column follows the
line-of-sight truncation: `truncation=:spherical` (default) counts only gas inside
`truncation_radius * R200c` (tau = 0 for θ >= X θ200c, the painting radius);
`truncation=:cylinder` restores the previous infinite line of sight with the projected
cut at `truncation_radius * θ200c`.
"""
struct BattagliaTauProfile{T,C,R<:AbstractTruncation} <: AbstractBattagliaTauProfile{T}
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
    BattagliaTauProfilePhysical(; ..., truncation=:spherical, truncation_radius=4.0)

Same as `BattagliaTauProfile`, but the radial coordinate is a physical length (Unitful) in
`rho_2d`, `ne2d` and `compute_tau`. `profile_grid` has no fast path for it (it tabulates the
direct values).
"""
struct BattagliaTauProfilePhysical{T,C,R<:AbstractTruncation} <: AbstractBattagliaTauProfile{T}
    f_b::T  # Omega_b / Omega_c = 0.0486 / 0.2589
    cosmo::C
    P0::PowerLawParam{T}
    x_c::PowerLawParam{T}
    alpha::PowerLawParam{T}
    beta::PowerLawParam{T}
    gamma::PowerLawParam{T}
    truncation::R
end

# positional constructors with the pre-truncation field list get the default truncation
BattagliaTauProfile(f_b, cosmo, P0, x_c, alpha, beta, gamma) =
    BattagliaTauProfile(f_b, cosmo, P0, x_c, alpha, beta, gamma, default_truncation())
BattagliaTauProfilePhysical(f_b, cosmo, P0, x_c, alpha, beta, gamma) =
    BattagliaTauProfilePhysical(f_b, cosmo, P0, x_c, alpha, beta, gamma, default_truncation())


function BattagliaTauProfile(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774,
        P0_amp::T=4.0e3, P0_alpha_m::T=0.29, P0_alpha_z::T=-0.66,
        x_c_amp::T=0.5, x_c_alpha_m::T=0.0, x_c_alpha_z::T=0.0,
        alpha_amp::T=0.88, alpha_alpha_m::T=-0.03, alpha_alpha_z::T=0.19,
        beta_amp::T=3.83, beta_alpha_m::T=0.04, beta_alpha_z::T=-0.025,
        gamma_amp::T=-0.2, gamma_alpha_m::T=0.0, gamma_alpha_z::T=0.0,
        truncation=:spherical, truncation_radius=DEFAULT_TRUNCATION_RADIUS) where {T <: Real}
    OmegaM=Omega_b+Omega_c
    f_b = Omega_b / OmegaM
    cosmo = get_cosmology(T, h=h, Neff=3.046, OmegaM=OmegaM)
    P0 = PowerLawParam(T(P0_amp), T(P0_alpha_m), T(P0_alpha_z))
    x_c = PowerLawParam(T(x_c_amp), T(x_c_alpha_m), T(x_c_alpha_z))
    alpha = PowerLawParam(T(alpha_amp), T(alpha_alpha_m), T(alpha_alpha_z))
    beta = PowerLawParam(T(beta_amp), T(beta_alpha_m), T(beta_alpha_z))
    gamma = PowerLawParam(T(gamma_amp), T(gamma_alpha_m), T(gamma_alpha_z))
    tr = make_truncation(truncation, truncation_radius)
    return BattagliaTauProfile{T, typeof(cosmo), typeof(tr)}(f_b, cosmo, P0, x_c, alpha, beta, gamma, tr)
end


function BattagliaTauProfilePhysical(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774,
        P0_amp::T=4.0e3, P0_alpha_m::T=0.29, P0_alpha_z::T=-0.66,
        x_c_amp::T=0.5, x_c_alpha_m::T=0.0, x_c_alpha_z::T=0.0,
        alpha_amp::T=0.88, alpha_alpha_m::T=-0.03, alpha_alpha_z::T=0.19,
        beta_amp::T=3.83, beta_alpha_m::T=0.04, beta_alpha_z::T=-0.025,
        gamma_amp::T=-0.2, gamma_alpha_m::T=0.0, gamma_alpha_z::T=0.0,
        truncation=:spherical, truncation_radius=DEFAULT_TRUNCATION_RADIUS) where {T <: Real}
    OmegaM=Omega_b+Omega_c
    f_b = Omega_b / OmegaM
    cosmo = get_cosmology(T, h=h, Neff=3.046, OmegaM=OmegaM)
    P0 = PowerLawParam(T(P0_amp), T(P0_alpha_m), T(P0_alpha_z))
    x_c = PowerLawParam(T(x_c_amp), T(x_c_alpha_m), T(x_c_alpha_z))
    alpha = PowerLawParam(T(alpha_amp), T(alpha_alpha_m), T(alpha_alpha_z))
    beta = PowerLawParam(T(beta_amp), T(beta_alpha_m), T(beta_alpha_z))
    gamma = PowerLawParam(T(gamma_amp), T(gamma_alpha_m), T(gamma_alpha_z))
    tr = make_truncation(truncation, truncation_radius)
    return BattagliaTauProfilePhysical{T, typeof(cosmo), typeof(tr)}(f_b, cosmo, P0, x_c, alpha, beta, gamma, tr)
end

truncation(model::Union{BattagliaTauProfile,BattagliaTauProfilePhysical}) = model.truncation
tabulates_chord_mean(model::BattagliaTauProfile) = model.truncation isa SphereTruncation

function get_params(model::AbstractBattagliaTauProfile{T}, M_200c, z) where T
	z₁ = z + 1
	m = M_200c / (1e14M_sun)
    P₀ = powerlaw_value(model.P0, m, z₁)
	α = powerlaw_value(model.alpha, m, z₁)
	β = powerlaw_value(model.beta, m, z₁)
	xc = powerlaw_value(model.x_c, m, z₁)
    γ = powerlaw_value(model.gamma, m, z₁)
    return (xc=T(xc), α=T(α), β=T(β), γ=T(γ), P₀=T(P₀))
end

# if angular, return the R200 size in radians
function object_size(model::BattagliaTauProfile{T,C}, physical_size, z) where {T,C}
    d_A = angular_diameter_dist(model.cosmo, z)
    phys_siz_unitless = T(ustrip(uconvert(unit(d_A), physical_size)))
    d_A_unitless = T(ustrip(d_A))
    return atan(phys_siz_unitless, d_A_unitless)
end

# if physical, return the R200 size in Mpc
function object_size(::BattagliaTauProfilePhysical{T,C}, physical_size, z) where {T,C}
    return physical_size
end

# returns a density, which we can check against Msun/Mpc²
# The dimensionless LOS kernel follows truncation(model): the finite chord inside the
# X R200c sphere (default), or the legacy infinite LOS for CylinderTruncation.
function rho_2d(model::AbstractBattagliaTauProfile, r, m200c, z)
    par = get_params(model, m200c, z)
    r200c = R_Δ(model, m200c, z, 200)   # this is physical units
    X = r / object_size(model, r200c, z)  # either ang/ang or phys/phys
    rho_crit = ρ_crit(model, z)   # this is physical units
    rho_fit = par.P₀ * los_kernel(truncation(model), X, par.xc, par.α, par.β, par.γ)   # dimensionless LOS kernel, see truncation.jl
    return rho_fit * model.f_b * rho_crit * r200c # mistake in battaglia 2016: need f_b to convert from m to gas. note r200c due to integral over X
end

# mean particle mass per electron for a fully ionized H + He gas with X_H = 0.76
function _mass_per_electron()
    me = constants.ElectronMass
    mH = constants.ProtonMass
    mHe = 4mH
    xH = 0.76
    nH_ne = 2xH / (xH + 1)
    nHe_ne = (1 - xH)/(2 * (1 + xH))
    return (me + nH_ne*mH + nHe_ne*mHe)
end

function ne2d(model::AbstractBattagliaTauProfile, r, m200c, z)
    factor = _mass_per_electron()
    result = rho_2d(model, r, m200c, z)  # (Msun) / (Mpc)^2
    return 0.9 * result / factor
end

# r is either a physical or angular radius. unitful does not do little h, so physical radius
# is always in Mpc, and angular radius is always in radians.
function compute_tau(model, r, m200c, z)
    return constants.ThomsonCrossSection * ne2d(model, r, m200c, z) + 0  # ensure unitless
end


# evaluation functions never take Unitful inputs, only raw numbers
# mass is in Msun
function (tau_model::AbstractBattagliaTauProfile)(r, m200c, z)
    return compute_tau(tau_model, r, m200c * M_sun, z)
end


# ---- profile_grid fast path for spherically truncated column profiles (tau here, DM in
# profiles_frb.jl). The grid tabulates the chord mean g = column / (2L); see truncation.jl.

# electron column per unit dimensionless LOS kernel: ne2d(model, r, M, z) = this * K(x)
function _ne2d_per_kernel(model::AbstractBattagliaTauProfile, par, m200c, z)
    r200c = R_Δ(model, m200c, z, 200)
    return 0.9 * (par.P₀ * model.f_b * ρ_crit(model, z) * r200c) / _mass_per_electron()
end

# cylinder: no fast path, profile_grid evaluates model(θ, M, z) exactly as before
_prepare_column_slice(::CylinderTruncation, model, mass, z) = nothing

function _prepare_column_slice(t::SphereTruncation, model::AbstractBattagliaTauProfile, mass, z)
    m = mass * M_sun
    par = get_params(model, m, z)
    r200c = R_Δ(model, m, z, 200)
    theta_scale = Float64(object_size(model, r200c, z))
    amplitude = Float64(_column_amplitude(model, par, m, z))
    return (; theta_scale, xc=par.xc, alpha=par.α, beta=par.β, gamma=par.γ, amplitude, truncation=t)
end

@inline function _evaluate_column_slice(p::NamedTuple, theta)
    x = theta / p.theta_scale
    return p.amplitude * grid_kernel(p.truncation, x, p.xc, p.alpha, p.beta, p.gamma)
end

# tau value per unit LOS kernel (dimensionless)
_column_amplitude(model::BattagliaTauProfile, par, m200c, z) =
    constants.ThomsonCrossSection * _ne2d_per_kernel(model, par, m200c, z) + 0

prepare_profile_slice(model::BattagliaTauProfile, mass, z) =
    _prepare_column_slice(model.truncation, model, mass, z)
@inline evaluate_profile_slice(::BattagliaTauProfile, p::NamedTuple, theta, mass, z) =
    _evaluate_column_slice(p, theta)
