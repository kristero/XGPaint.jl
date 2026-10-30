abstract type AbstractFRBProfile{T} <: AbstractBattagliaTauProfile{T} end

const M2_TO_PC_CM3 = ustrip(u"pc*cm^-3", 1u"m^-2")


"""
    HaloDMProfile(; Omega_c=0.2589, Omega_b=0.0486, h=0.6774, P0_amp=4.0e3, ...,
        truncation=:spherical, truncation_radius=4.0)
    HaloDMProfile(tau_model::BattagliaTauProfile)

Observed halo dispersion measure (pc/cm^3) from the Battaglia (2016) gas-density fit,
`model(θ, M200c, z)`. The electron column uses the same line-of-sight truncation keywords as
`BattagliaTauProfile` (spherical by default: DM = 0 for θ >= X θ200c);
`HaloDMProfile(tau_model)` copies the parameters and the truncation of `tau_model`.
`BattagliaFRBProfile` is an alias.
"""
struct HaloDMProfile{T,C,R<:AbstractTruncation} <: AbstractFRBProfile{T}
    f_b::T
    cosmo::C
    P0::PowerLawParam{T}
    x_c::PowerLawParam{T}
    alpha::PowerLawParam{T}
    beta::PowerLawParam{T}
    gamma::PowerLawParam{T}
    truncation::R
end

# positional constructor with the pre-truncation field list gets the default truncation
HaloDMProfile(f_b, cosmo, P0, x_c, alpha, beta, gamma) =
    HaloDMProfile(f_b, cosmo, P0, x_c, alpha, beta, gamma, default_truncation())

function HaloDMProfile(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774,
        P0_amp::T=4.0e3, P0_alpha_m::T=0.29, P0_alpha_z::T=-0.66,
        x_c_amp::T=0.5, x_c_alpha_m::T=0.0, x_c_alpha_z::T=0.0,
        alpha_amp::T=0.88, alpha_alpha_m::T=-0.03, alpha_alpha_z::T=0.19,
        beta_amp::T=3.83, beta_alpha_m::T=0.04, beta_alpha_z::T=-0.025,
        gamma_amp::T=-0.2, gamma_alpha_m::T=0.0, gamma_alpha_z::T=0.0,
        truncation=:spherical, truncation_radius=DEFAULT_TRUNCATION_RADIUS) where {T <: Real}
    OmegaM = Omega_b + Omega_c
    f_b = Omega_b / OmegaM
    cosmo = get_cosmology(T, h=h, Neff=3.046, OmegaM=OmegaM)
    P0 = PowerLawParam(T(P0_amp), T(P0_alpha_m), T(P0_alpha_z))
    x_c = PowerLawParam(T(x_c_amp), T(x_c_alpha_m), T(x_c_alpha_z))
    alpha = PowerLawParam(T(alpha_amp), T(alpha_alpha_m), T(alpha_alpha_z))
    beta = PowerLawParam(T(beta_amp), T(beta_alpha_m), T(beta_alpha_z))
    gamma = PowerLawParam(T(gamma_amp), T(gamma_alpha_m), T(gamma_alpha_z))
    tr = make_truncation(truncation, truncation_radius)
    return HaloDMProfile{T, typeof(cosmo), typeof(tr)}(f_b, cosmo, P0, x_c, alpha, beta, gamma, tr)
end

function HaloDMProfile(tau_model::BattagliaTauProfile{T,C,R}) where {T,C,R}
    return HaloDMProfile{T,C,R}(tau_model.f_b, tau_model.cosmo, tau_model.P0, tau_model.x_c,
        tau_model.alpha, tau_model.beta, tau_model.gamma, tau_model.truncation)
end

BattagliaFRBProfile(; kwargs...) = HaloDMProfile(; kwargs...)

truncation(model::HaloDMProfile) = model.truncation
tabulates_chord_mean(model::HaloDMProfile) = model.truncation isa SphereTruncation

function object_size(model::HaloDMProfile{T,C}, physical_size, z) where {T,C}
    d_A = angular_diameter_dist(model.cosmo, z)
    phys_siz_unitless = T(ustrip(uconvert(unit(d_A), physical_size)))
    d_A_unitless = T(ustrip(d_A))
    return atan(phys_siz_unitless, d_A_unitless)
end

"""
    compute_DM(model, r, M_200c, z)

Compute the observed halo DM contribution in `pc/cm^3` directly from the
projected electron column density (line of sight as set by `truncation(model)`).
"""
function compute_DM(model::AbstractFRBProfile{T}, r, M_200c, z) where T
    electron_column = ustrip(u"m^-2", ne2d(model, r, M_200c, z))
    return T(electron_column * M2_TO_PC_CM3 / (one(z) + z))
end

# direct evaluation follows the rest of the profile API: mass is in Msun and
# the returned value is a plain number in pc/cm^3
(model::AbstractFRBProfile)(r, M_200c, z) = compute_DM(model, r, M_200c * M_sun, z)

# profile_grid fast path (spherical truncation): DM per unit dimensionless LOS kernel
_column_amplitude(model::HaloDMProfile, par, m200c, z) =
    ustrip(u"m^-2", _ne2d_per_kernel(model, par, m200c, z)) * M2_TO_PC_CM3 / (one(z) + z)

prepare_profile_slice(model::HaloDMProfile, mass, z) =
    _prepare_column_slice(model.truncation, model, mass, z)
@inline evaluate_profile_slice(::HaloDMProfile, p::NamedTuple, theta, mass, z) =
    _evaluate_column_slice(p, theta)
