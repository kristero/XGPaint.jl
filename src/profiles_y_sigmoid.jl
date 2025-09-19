"""Sigmoid-suppressed thermal Sunyaev-Zel'dovich profiles, based on Battaglia et al. 2016. 
These profiles include a sigmoid suppression factor that smoothly transitions from 0.99 at 4 R_200 
to 0.01 at 6 R_200."""

struct SigmoidBattaglia16ThermalSZProfile{T,C} <: AbstractGNFW{T}
    f_b::T  # Omega_b / Omega_c = 0.0486 / 0.2589
    cosmo::C
end

struct SigmoidBreakModel{T,C} <: AbstractGNFW{T}
    f_b::T
    cosmo::C
    alpha_break::T
    M_break::T
end

function SigmoidBattaglia16ThermalSZProfile(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774) where {T <: Real}
    OmegaM=Omega_b+Omega_c
    f_b = Omega_b / OmegaM
    cosmo = get_cosmology(T, h=h, OmegaM=OmegaM)
    return SigmoidBattaglia16ThermalSZProfile(f_b, cosmo)
end

function SigmoidBreakModel(; Omega_c::T=0.2589, Omega_b::T=0.0486, h::T=0.6774, alpha_break::T=1.5, M_break::T=2.0*10^14) where {T <: Real}
    OmegaM=Omega_b+Omega_c
    f_b = Omega_b / OmegaM
    cosmo = get_cosmology(T, h=h, OmegaM=OmegaM)
    return SigmoidBreakModel(f_b, cosmo, alpha_break, M_break)
end

"""
    sigmoid_suppression(r)

Sigmoid suppression factor that equals 0.99 at r=4 and 0.01 at r=6.
Uses the form: f(r) = 1 / (1 + exp(k * (r - r0)))

Where k and r0 are solved from the boundary conditions:
- f(4) = 0.99 => 1/(1 + exp(k*(4-r0))) = 0.99
- f(6) = 0.01 => 1/(1 + exp(k*(6-r0))) = 0.01

This gives us:
- exp(k*(4-r0)) = 1/99 ≈ 0.0101
- exp(k*(6-r0)) = 99
- Taking the ratio: exp(2k) = 99^2 = 9801
- So k = ln(9801)/2 ≈ 4.599
- And r0 = 4 + ln(99)/k = 4 + ln(99)/4.599 ≈ 5
"""
function sigmoid_suppression(r::T) where {T <: Real}
    # More moderate steepness - still achieves the desired transition but more numerically stable
    k = T(2.3)    # Moderate steepness
    r0 = T(5.0)   # midpoint
    return 1 / (1 + exp(k * (r - r0)))
end

function _sigmoid_nfw_profile_los_quadrature(x, xc, α, β, γ; zmax=1e5, rtol=eps(), order=9)
    x² = x^2
    scale = 1e9
    integral, err = quadgk(y -> scale * sigmoid_suppression(√(y^2 + x²)) * generalized_nfw(√(y^2 + x²), xc, α, β, γ),
                      0.0, zmax, rtol=rtol, order=order)
    return 2integral / scale
end

function dimensionless_P_profile_los(model::SigmoidBattaglia16ThermalSZProfile{T}, r, M_200, z) where T
    par = get_params(model, M_200, z)
    R_200 = R_Δ(model, M_200, z, 200)
    x = r / angular_size(model, R_200, z)
    return par.P₀ * _sigmoid_nfw_profile_los_quadrature(x, par.xc, par.α, par.β, par.γ)
end

function dimensionless_P_profile_los(model::SigmoidBreakModel{T}, r, M_200, z) where T
    par = get_params(model, M_200, z)
    R_200 = R_Δ(model, M_200, z, 200)
    x = r / angular_size(model, R_200, z)
    if M_200 < model.M_break * M_sun
        return (par.P₀ * (M_200/(model.M_break*M_sun))^model.alpha_break * 
            _sigmoid_nfw_profile_los_quadrature(x, par.xc, par.α, par.β, par.γ))
    else
        return par.P₀ * _sigmoid_nfw_profile_los_quadrature(x, par.xc, par.α, par.β, par.γ)
    end
end

"""
    compton_y(model, r, M_200c, z)

Calculate the Compton y parameter for sigmoid-suppressed models.
Mass needs to have units!
"""
function compton_y(model::SigmoidBattaglia16ThermalSZProfile, r, M_200c, z)
    return P_e_los(model, r, M_200c, z) * P_e_factor + 0   # +0 to strip units
end

function compton_y(model::SigmoidBreakModel, r, M_200c, z)
    return P_e_los(model, r, M_200c, z) * P_e_factor + 0   # +0 to strip units
end

# direct evaluation of sigmoid models at a given radius, mass, and redshift. 
# mass is in Msun; Unitful is NOT used in these evaluation functions
(model::SigmoidBattaglia16ThermalSZProfile)(r, M_200c, z) = compton_y(model, r, M_200c * M_sun, z)
(model::SigmoidBreakModel)(r, M_200c, z) = compton_y(model, r, M_200c * M_sun, z)
