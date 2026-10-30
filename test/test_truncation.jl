# Tests of the line-of-sight truncation (src/truncation.jl): spherical X R200c truncation is
# the default for every gNFW profile; truncation=:cylinder reproduces the previous outputs.
using XGPaint
using XGPaint: M_sun, SphereTruncation, CylinderTruncation, generalized_nfw
using Test
using Pixell
using Healpix
using Unitful, UnitfulAstro

const tr_quadgk = XGPaint.QuadGK.quadgk
const TR_COSMO = (Omega_c=0.261, Omega_b=0.049, h=0.68)
const TR_X = 4.0

tr_theta200(model, M, z) = XGPaint.compute_θmax(model, M * M_sun, z; mult=1)
tr_shape(par) = (Float64(par.xc), Float64(par.α), Float64(par.β), Float64(par.γ))

# independent reference: brute-force QuadGK of the 3-D shape along the chord |l| <= L
function tr_ref_column(x, xc, α, β, γ, X)
    x >= X && return 0.0
    L = sqrt(X^2 - x^2)
    f(l) = generalized_nfw(hypot(x, l), xc, α, β, γ)
    pts = sort(unique(filter(p -> 0 < p < L, [x, 10x, 1.0])))
    return 2 * tr_quadgk(f, 0.0, pts..., L; rtol=1e-12)[1]
end
# x = 0: substitute l = s^k, k = 1/(1+γ), which removes the l^γ cusp
function tr_ref_center(xc, α, β, γ, X)
    k = 1 / (1 + γ)
    f(s) = generalized_nfw(s^k, xc, α, β, γ) * k * s^(k - 1)
    return 2 * tr_quadgk(f, 0.0, X^(1 + γ); rtol=1e-12)[1]
end
# Y of the sphere: projected (2π ∫ x K(x) dx over the disc) and volume (4π ∫ r^2 p dr)
tr_Y_projected(t::SphereTruncation, xc, α, β, γ) = tr_quadgk(
    x -> 2π * x * XGPaint.los_kernel(t, x, xc, α, β, γ), 0.0, 0.1, 1.0, t.X; rtol=1e-10)[1]
tr_Y_volume(xc, α, β, γ, X) = tr_quadgk(
    r -> 4π * r^2 * generalized_nfw(r, xc, α, β, γ), 0.0, 0.1, 1.0, X; rtol=1e-12)[1]

struct TruncTestDummyProfile{T} <: XGPaint.AbstractProfile{T}
    value::T
end
(m::TruncTestDummyProfile)(θ, Mh, z) = m.value

@testset "line-of-sight truncation" begin

    tsz = XGPaint.Battaglia16ThermalSZProfile(; TR_COSMO...)
    tsz_cyl = XGPaint.Battaglia16ThermalSZProfile(; TR_COSMO..., truncation=:cylinder)
    brk = XGPaint.BreakModel(; TR_COSMO...)
    brk_cyl = XGPaint.BreakModel(; TR_COSMO..., truncation=:cylinder)
    tau = XGPaint.BattagliaTauProfile(; TR_COSMO...)
    tau_cyl = XGPaint.BattagliaTauProfile(; TR_COSMO..., truncation=:cylinder)
    dm = XGPaint.HaloDMProfile(; TR_COSMO...)
    dm_cyl = XGPaint.HaloDMProfile(; TR_COSMO..., truncation=:cylinder)

    @testset "API and defaults" begin
        @test XGPaint.truncation(tsz) === SphereTruncation(4.0)
        @test XGPaint.truncation(XGPaint.Battaglia16ThermalSZProfile()) isa SphereTruncation
        @test XGPaint.truncation(tsz_cyl) === CylinderTruncation(4.0, 1e5)
        for m in (brk, tau, dm, XGPaint.BattagliaTauProfilePhysical(), XGPaint.BattagliaFRBProfile(),
                  XGPaint.RSZPerturbativeProfile())
            @test XGPaint.truncation(m) === SphereTruncation(4.0)
        end
        @test XGPaint.truncation(XGPaint.BattagliaFRBProfile(truncation=:cylinder)) isa CylinderTruncation
        @test XGPaint.truncation(XGPaint.Battaglia16ThermalSZProfile(truncation_radius=3.0)) === SphereTruncation(3.0)
        @test XGPaint.truncation(XGPaint.Battaglia16ThermalSZProfile(truncation=SphereTruncation(2.5))) === SphereTruncation(2.5)
        @test XGPaint.truncation(XGPaint.HaloDMProfile(tau_cyl)) === XGPaint.truncation(tau_cyl)
        @test XGPaint.truncation(XGPaint.HaloDMProfile(tau)) === SphereTruncation(4.0)
        @test XGPaint.make_truncation(:sphere, 3.0) === SphereTruncation(3.0)
        for s in (:cylinder, :cylindrical, :projected, :infinite, :legacy, "cylinder")
            @test XGPaint.make_truncation(s) === CylinderTruncation(4.0, 1e5)
        end
        @test_throws ArgumentError XGPaint.make_truncation(:bogus)
        @test_throws ArgumentError SphereTruncation(0.0)
        @test_throws ArgumentError SphereTruncation(Inf)
        @test_throws ArgumentError CylinderTruncation(-1.0)
        # positional constructor with the old field list gets the default
        @test XGPaint.truncation(XGPaint.Battaglia16ThermalSZProfile(tsz.f_b, tsz.cosmo, tsz.P0,
            tsz.x_c, tsz.alpha, tsz.beta, tsz.gamma)) === SphereTruncation(4.0)
        # models that carry no truncation keep the legacy painter behaviour
        dummy = TruncTestDummyProfile(1.0)
        @test XGPaint.truncation(dummy) === CylinderTruncation(4.0, 1e5)
        @test XGPaint.truncation_radius(dummy) == 4.0
        @test !XGPaint.tabulates_chord_mean(dummy)
        @test XGPaint.paint_value(dummy, 0.1, 0.1, 0.2, 1e14, 0.5) == 1.0
        @test XGPaint.tabulates_chord_mean(tsz) && XGPaint.tabulates_chord_mean(tau) && XGPaint.tabulates_chord_mean(dm)
        @test !XGPaint.tabulates_chord_mean(tsz_cyl) && !XGPaint.tabulates_chord_mean(dm_cyl)
        @test !XGPaint.tabulates_chord_mean(XGPaint.BattagliaTauProfilePhysical())
        # θmax = X θ200c
        for (M, z) in ((1e13, 0.2), (1e15, 1.5))
            t200 = tr_theta200(tsz, M, z)
            @test XGPaint.compute_θmax(tsz, M * M_sun, z) == 4 * t200
            @test XGPaint.compute_θmax(tsz_cyl, M * M_sun, z) == 4 * t200
            @test XGPaint.compute_θmax(XGPaint.Battaglia16ThermalSZProfile(; TR_COSMO..., truncation_radius=2.5),
                M * M_sun, z) ≈ 2.5 * t200 rtol=1e-15
            @test XGPaint.compute_θmax(tsz, M * M_sun, z; mult=6) ≈ 6 * t200 rtol=1e-15
        end
    end

    @testset "chord quadrature vs independent reference integral" begin
        trunc = SphereTruncation(TR_X)
        for mdl in (tsz, tau), (M, z) in ((1e12, 0.01), (1e14, 0.5), (3e15, 2.0))
            xc, α, β, γ = tr_shape(XGPaint.get_params(mdl, M * M_sun, z))
            for x in (1e-4, 1e-2, 0.3, 1.0, 2.5, 3.9, 3.999)
                @test XGPaint.los_kernel(trunc, x, xc, α, β, γ) ≈ tr_ref_column(x, xc, α, β, γ, TR_X) rtol=1e-9
            end
            @test XGPaint.los_kernel(trunc, 0.0, xc, α, β, γ) ≈ tr_ref_center(xc, α, β, γ, TR_X) rtol=1e-9
            @test XGPaint.los_kernel(trunc, TR_X, xc, α, β, γ) == 0.0
            @test XGPaint.los_kernel(trunc, 5.0, xc, α, β, γ) == 0.0
            # projected Y inside the sphere equals the volume integral 4π ∫_0^X r^2 p dr
            @test tr_Y_projected(trunc, xc, α, β, γ) ≈ tr_Y_volume(xc, α, β, γ, TR_X) rtol=1e-8
            # the legacy kernel is untouched
            @test XGPaint.los_kernel(CylinderTruncation(), 0.7, xc, α, β, γ) ==
                  XGPaint._nfw_profile_los_quadrature(0.7, xc, α, β, γ)
        end
        @test_throws ErrorException XGPaint.chord_quadrature(0.0, 0.5, 1.0, 4.0, -1.2, 3.0)
        @test XGPaint.chord_quadrature(0.5, 0.5, 1.0, 4.0, -0.3, 0.0) == 0.0
    end

    @testset "behaviour at the sphere edge x -> X" begin
        xc, α, β, γ = tr_shape(XGPaint.get_params(tsz, 1e14 * M_sun, 0.5))
        pX = generalized_nfw(TR_X, xc, α, β, γ)
        g(x) = XGPaint.chord_mean(x, xc, α, β, γ, TR_X)
        K(x) = XGPaint.los_kernel(SphereTruncation(TR_X), x, xc, α, β, γ)
        @test abs(g(TR_X * (1 - 1e-9)) / pX - 1) < 1e-6
        @test g(TR_X) ≈ pX rtol=1e-14
        gs = [g(TR_X * f) for f in (1.001, 1.5, 10.0, 1e4)]
        @test all(isfinite, gs) && all(>(0), gs) && issorted(gs; rev=true)
        @test g(TR_X * 0.999) > g(TR_X) > g(TR_X * 1.001)
        # the smooth exterior is the analytic continuation of the chord mean
        @test XGPaint.smooth_exterior_mean(3.0, xc, α, β, γ, TR_X) ≈ g(3.0) rtol=1e-6
        @test XGPaint.smooth_exterior_mean(TR_X, xc, α, β, γ, TR_X) ≈ pX rtol=1e-14
        # y -> 2 L p(X): falls like sqrt(X - x)
        @test K(TR_X * (1 - 1e-6)) / K(TR_X * (1 - 1e-4)) ≈ 0.1 rtol=1e-2
        @test K(TR_X * (1 - 1e-6)) ≈ 2 * sqrt(TR_X^2 - (TR_X * (1 - 1e-6))^2) * pX rtol=1e-5
        θmax = 1.3e-3
        @test XGPaint.chord_factor(θmax * (1 + 1e-12), θmax, TR_X) == 0.0
        @test XGPaint.chord_factor(θmax, θmax, TR_X) == 0.0
        @test XGPaint.chord_factor(0.0, θmax, TR_X) == 2TR_X
        @test XGPaint.chord_factor(0.5θmax, θmax, TR_X) ≈ 2 * sqrt(TR_X^2 - (TR_X / 2)^2) rtol=1e-15
    end

    @testset "direct profiles: bounds, monotonicity, zero outside" begin
        for (sph, cyl) in ((tsz, tsz_cyl), (brk, brk_cyl), (tau, tau_cyl), (dm, dm_cyl))
            for (M, z) in ((5e13, 0.4), (2e15, 0.1))   # below and above the BreakModel break
                t200 = tr_theta200(sph, M, z)
                xs = (0.05, 0.5, 1.0, 2.0, 3.0, 3.9, 3.999)
                ys = [sph(x * t200, M, z) for x in xs]
                yc = [cyl(x * t200, M, z) for x in xs]
                @test all(>(0), ys)
                @test all(ys .<= yc .* (1 + 1e-9))
                @test issorted(ys; rev=true)
                @test sph(4.0 * t200 * (1 + 1e-12), M, z) == 0
                @test sph(4.5 * t200, M, z) == 0
                @test cyl(4.5 * t200, M, z) > 0
                θmax = XGPaint.compute_θmax(sph, M * M_sun, z)
                @test sph(θmax * (1 + 1e-12), M, z) == 0
            end
        end
    end

    @testset "match the HalfDome production engine; :cylinder reproduces the previous outputs" begin
        # spherical_value_direct of HalfDome tSZ_64k_flat_prior_SO_baseline_deproj0/engine
        # (ChordMeanProfile, X = 4) and XGPaint HEAD 1660e08 values, at θ = x θ200c for
        # (M, z, x) = (1e14, 0.5, 0.5), (1e13, 1.0, 2.0), (1e15, 0.1, 3.5)
        cases = ((1e14, 0.5), (1e13, 1.0), (1e15, 0.1))
        θ_tsz = (0.00031438541800625356, 0.000364882542266243, 0.01820762816637885)
        θ_tau = (0.0003143854124766215, 0.000364882528372885, 0.018207628116500618)
        prod = Dict(
            "tsz" => (1.100324480512151e-6, 6.425256533289549e-9, 1.4737928158481897e-8),
            "tau" => (0.0005191647913943066, 2.7683954304133775e-5, 8.888782491574239e-6),
            "dm" => (168.60918862383323, 6.74318995214868, 3.936559858697541))
        head = Dict(
            "tsz" => (1.1013040120868205e-6, 6.6711638856810726e-9, 2.2620603717565692e-8),
            "break" => (3.893697675472712e-7, 7.458587968712258e-11, 2.2620603717565692e-8),
            "tau" => (0.0005234951791512895, 3.153395186829769e-5, 1.5853297257031282e-5),
            "dm" => (170.01556898366184, 7.680962952539396, 7.020922569450404))
        sphs = Dict("tsz" => tsz, "tau" => tau, "dm" => dm)
        cyls = Dict("tsz" => tsz_cyl, "break" => brk_cyl, "tau" => tau_cyl, "dm" => dm_cyl)
        θs = Dict("tsz" => θ_tsz, "break" => θ_tsz, "tau" => θ_tau, "dm" => θ_tau)
        for (k, mdl) in sphs, (i, (M, z)) in enumerate(cases)
            @test mdl(θs[k][i], M, z) ≈ prod[k][i] rtol=1e-10
        end
        for (k, mdl) in cyls, (i, (M, z)) in enumerate(cases)
            @test mdl(θs[k][i], M, z) ≈ head[k][i] rtol=1e-13
        end
        # profile grids: HEAD (cylinder) and production ChordMeanProfile (sphere) checksums
        lt = LinRange(log(1e-6), log(1e-1), 24); zs = [0.1, 0.5, 1.0]; lms = [13.0, 14.0, 15.0]
        @test sum(XGPaint.profile_grid(tsz_cyl, lt, zs, lms)[4]) ≈ 0.005269833292116343 rtol=1e-13
        @test sum(XGPaint.profile_grid(tau_cyl, lt, zs, lms)[4]) ≈ 0.43901787527934233 rtol=1e-13
        @test XGPaint.profile_grid(dm_cyl, lt, zs, lms)[4][5, 2, 3] ≈ 2954.699356961661 rtol=1e-13
        @test sum(XGPaint.profile_grid(tsz, lt, zs, lms)[4]) ≈ 0.0006588858295648848 rtol=1e-10
        @test sum(XGPaint.profile_grid(tau, lt, zs, lms)[4]) ≈ 0.05483456392508631 rtol=1e-10
        @test XGPaint.profile_grid(dm, lt, zs, lms)[4][5, 2, 3] ≈ 369.075040783994 rtol=1e-10
        # painted CAR map of the HEAD direct model (3 halos, 0.5 arcmin)
        box = [0.6 -0.6; -0.6 0.6] * Pixell.degree
        shape, wcs = geometry(Pixell.CarClenshawCurtis{Float64}, box, 0.5 * Pixell.arcminute)
        ws = profileworkspace(shape, wcs)
        masses = [1e15, 3e14, 1e14]; reds = [0.3, 0.5, 0.2]
        ras = [0.0, 0.2, -0.25] .* Pixell.degree; decs = [0.0, -0.15, 0.3] .* Pixell.degree
        m = Enmap(zeros(shape), wcs)
        paint!(m, ws, tsz_cyl, masses, reds, ras, decs)
        @test sum(m.data) ≈ 0.006940097354406552 rtol=1e-12
        @test maximum(m.data) ≈ 0.00011948151767020965 rtol=1e-12
        @test count(!iszero, m.data) == 11291
        m = Enmap(zeros(shape), wcs)
        paint!(m, ws, tau_cyl, masses, reds, ras, decs, [0.001, -0.002, 0.0005])
        @test sum(m.data) ≈ 0.0008110295526134219 rtol=1e-12
    end

    @testset "profile_grid tabulates a positive chord mean" begin
        lt = LinRange(log(1e-6), log(0.05), 32); zs = [0.05, 0.3, 0.8, 2.0]; lms = [12.5, 13.5, 14.5, 15.5]
        for (sph, cyl) in ((tsz, tsz_cyl), (brk, brk_cyl), (tau, tau_cyl), (dm, dm_cyl))
            A = XGPaint.profile_grid(sph, lt, zs, lms)[4]
            @test all(isfinite, A) && all(>(0), A)
            ninside = 0
            for (im, lm) in enumerate(lms), (iz, z) in enumerate(zs), (it, l) in enumerate(lt)
                M = 10^lm; θ = exp(l)
                θmax = XGPaint.compute_θmax(sph, M * M_sun, z)
                if θ < θmax
                    ninside += 1
                    @test A[it, iz, im] * XGPaint.chord_factor(θ, θmax, 4.0) ≈ sph(θ, M, z) rtol=1e-11
                end
            end
            @test ninside > 100
        end
    end

    @testset "spherical interpolator" begin
        kw = (N_logtheta=256, pad=64, N_z=6, N_logM=5, z_min=0.1, z_max=0.6, logM_min=13.0,
              logM_max=15.0, verbose=false)
        ip = build_interpolator(tsz; kw...)
        ip_cyl = build_interpolator(tsz_cyl; kw...)
        @test XGPaint.truncation(ip) === SphereTruncation(4.0)
        @test XGPaint.truncation(ip_cyl) isa CylinderTruncation
        # on the (M, z) nodes: only the log θ spline enters
        for M in (1e13, 10^14.5), z in (0.2, 0.5)
            t200 = tr_theta200(tsz, M, z)
            θmax = XGPaint.compute_θmax(tsz, M * M_sun, z)
            for x in (0.01, 0.1, 0.7, 2.0, 3.5, 3.95, 3.999)
                θ = x * t200
                @test ip(θ, M, z) ≈ tsz(θ, M, z) rtol=1e-4
                @test XGPaint.paint_value(ip, θ, θ, θmax, M, z) ≈ ip(θ, M, z) rtol=1e-14
            end
            @test ip(θmax * (1 + 1e-9), M, z) == 0
            @test XGPaint.paint_value(ip, 1.01θmax, 1.01θmax, θmax, M, z) == 0
            @test XGPaint.cached_value(ip, 1.01θmax, M, z) > 0
        end
        # off the nodes the spherical cache is as accurate as the legacy one
        err_s = 0.0; err_c = 0.0
        for M in (2.3e13, 4.1e14), z in (0.13, 0.47), x in (0.05, 0.5, 2.0, 3.9)
            θ = x * tr_theta200(tsz, M, z)
            err_s = max(err_s, abs(ip(θ, M, z) / tsz(θ, M, z) - 1))
            err_c = max(err_c, abs(ip_cyl(θ, M, z) / tsz_cyl(θ, M, z) - 1))
        end
        @test err_s <= 1.25 * err_c + 1e-6
    end

    @testset "painting: sphere edge, Y conservation, interpolator vs direct" begin
        M, z = 1e15, 0.3
        ip = build_interpolator(tsz; N_logtheta=256, pad=64, N_z=6, N_logM=5, z_min=0.1,
            z_max=0.6, logM_min=13.0, logM_max=15.0, verbose=false)
        t200 = tr_theta200(tsz, M, z)
        θmax = XGPaint.compute_θmax(tsz, M * M_sun, z)
        # analytic Y of the sphere (sr): A θ200^2 4π ∫_0^X r^2 p dr
        prep = XGPaint.prepare_profile_slice(tsz, M, z)
        Y = prep.amplitude * t200^2 * tr_Y_volume(prep.xc, prep.alpha, prep.beta, prep.gamma, TR_X)

        res_arcmin = 0.25
        box = [0.6 -0.6; -0.6 0.6] * Pixell.degree
        shape, wcs = geometry(Pixell.CarClenshawCurtis{Float64}, box, res_arcmin * Pixell.arcminute)
        ws = profileworkspace(shape, wcs)
        α_map, δ_map = posmap(shape, wcs)
        Ωpix = (res_arcmin * Pixell.arcminute)^2 .* cos.(δ_map)
        θpix = acos.(clamp.(1 .- ((cos.(δ_map) .* cos.(α_map) .- 1).^2 .+ (cos.(δ_map) .* sin.(α_map)).^2 .+
            sin.(δ_map).^2) ./ 2, -1, 1))
        m_ip = Enmap(zeros(shape), wcs); paint!(m_ip, ws, ip, [M], [z], [0.0], [0.0])
        m_dir = Enmap(zeros(shape), wcs); paint!(m_dir, ws, tsz, [M], [z], [0.0], [0.0])
        @test 2θmax < 1.2 * Pixell.degree   # disc fits in the patch
        @test sum(m_ip.data .* Ωpix) ≈ Y rtol=1e-2
        @test sum(m_dir.data .* Ωpix) ≈ Y rtol=1e-2
        outside = θpix .>= θmax * (1 + 1e-9)
        @test all(iszero, m_ip.data[outside]) && all(iszero, m_dir.data[outside])
        @test count(!iszero, m_ip.data) == count(θpix .< θmax)
        @test count(!iszero, m_dir.data) == count(θpix .< θmax)
        inner = θpix .< 3.9 * t200
        @test maximum(abs.(m_ip.data[inner] ./ m_dir.data[inner] .- 1)) < 1e-3
        @test maximum(abs.(m_ip.data .- m_dir.data)) < 1e-3 * maximum(m_dir.data)

        # kSZ (velocity) path with a spherical tau interpolator
        ip_tau = build_interpolator(tau; N_logtheta=256, pad=64, N_z=6, N_logM=5, z_min=0.1,
            z_max=0.6, logM_min=13.0, logM_max=15.0, verbose=false)
        m_k = Enmap(zeros(shape), wcs); paint!(m_k, ws, ip_tau, [M], [z], [0.0], [0.0], [-0.002])
        m_t = Enmap(zeros(shape), wcs); paint!(m_t, ws, ip_tau, [M], [z], [0.0], [0.0])
        m_td = Enmap(zeros(shape), wcs); paint!(m_td, ws, tau, [M], [z], [0.0], [0.0])
        @test m_k.data ≈ -0.002 .* m_t.data rtol=1e-12
        @test all(iszero, m_t.data[outside]) && all(iszero, m_td.data[outside])
        @test maximum(abs.(m_t.data .- m_td.data)) < 1e-3 * maximum(m_td.data)

        # HEALPix ring painter vs brute force, direct spherical model and interpolator
        Mh, zh = 1e15, 0.1
        θmaxh = XGPaint.compute_θmax(tsz, Mh * M_sun, zh)
        nside = 512
        res = Healpix.Resolution(nside)
        hws = XGPaint.HealpixRingProfileWorkspace(res)
        α₀, δ₀ = 0.7, 0.2
        hp_ip = HealpixMap{Float64,RingOrder}(zeros(nside2npix(nside)))
        hp_dir = HealpixMap{Float64,RingOrder}(zeros(nside2npix(nside)))
        hp_ref = HealpixMap{Float64,RingOrder}(zeros(nside2npix(nside)))
        paint!(hp_ip, hws, ip, [Mh], [zh], [α₀], [δ₀])
        paint!(hp_dir, hws, tsz, [Mh], [zh], [α₀], [δ₀])
        x₀, y₀, z₀ = ang2vec(π / 2 - δ₀, α₀)
        θmin = XGPaint.compute_θmin(tsz)
        ninside = 0
        for pix in eachindex(hp_ref.pixels)
            x₁, y₁, z₁ = pix2vecRing(res, pix)
            θ = acos(clamp(1 - ((x₁ - x₀)^2 + (y₁ - y₀)^2 + (z₁ - z₀)^2) / 2, -1.0, 1.0))
            if max(θmin, θ) < θmaxh
                ninside += 1
                hp_ref.pixels[pix] = tsz(max(θmin, θ), Mh, zh)
            end
        end
        @test ninside > 100
        @test hp_dir.pixels ≈ hp_ref.pixels rtol=1e-12
        @test count(!iszero, hp_dir.pixels) == ninside
        @test count(!iszero, hp_ip.pixels) == ninside
        @test maximum(abs.(hp_ip.pixels .- hp_ref.pixels)) < 1e-3 * maximum(hp_ref.pixels)
    end

    @testset "interpolator cache records and checks the truncation" begin
        kw = (N_logtheta=64, pad=16, N_z=4, N_logM=4, z_min=0.1, z_max=0.6, logM_min=13.0,
              logM_max=15.0, verbose=false)
        mktempdir() do dir
            f = joinpath(dir, "sph.jld2")
            ip1 = build_interpolator(tsz; cache_file=f, kw...)
            ip2 = build_interpolator(tsz; cache_file=f, overwrite=false, kw...)
            @test ip2(1e-3, 1e14, 0.3) == ip1(1e-3, 1e14, 0.3)
            saved = XGPaint.FileIO.load(f)
            @test saved["truncation"] == "spherical" && saved["truncation_radius"] == 4.0
            @test saved["tabulated_quantity"] == "chord_mean"
            @test_throws ErrorException build_interpolator(tsz_cyl; cache_file=f, overwrite=false, kw...)
            @test_throws ErrorException build_interpolator(
                XGPaint.Battaglia16ThermalSZProfile(; TR_COSMO..., truncation_radius=3.0);
                cache_file=f, overwrite=false, kw...)
            # a cache written before the truncation metadata existed is a legacy cylinder table
            g = joinpath(dir, "legacy.jld2")
            lt, zs, lms, A = XGPaint.profile_grid(tsz_cyl; N_logtheta=64, N_z=4, N_logM=4,
                z_min=0.1, z_max=0.6, logM_min=13.0, logM_max=15.0)
            XGPaint.FileIO.save(g, Dict("prof_logthetas" => lt, "prof_redshift" => zs,
                "prof_logMs" => lms, "prof_y" => A))
            ipl = build_interpolator(tsz_cyl; cache_file=g, overwrite=false, kw...)
            @test XGPaint.truncation(ipl) isa CylinderTruncation
            @test_throws ErrorException build_interpolator(tsz; cache_file=g, overwrite=false, kw...)
        end
    end

    @testset "SZ wrappers follow the y-profile truncation" begin
        # relativistic tSZ (perturbative): spherical y, painted with the fixed CAR painter
        rsz = XGPaint.RSZPerturbativeProfile(; TR_COSMO...)
        rsz_cyl = XGPaint.RSZPerturbativeProfile(; TR_COSMO..., truncation=:cylinder)
        M, z = 3e14, 0.4
        t200 = tr_theta200(rsz, M, z)
        @test XGPaint.compute_θmax(rsz, M * M_sun, z) ≈ 4 * t200 rtol=1e-15
        v_in = rsz(2.0 * t200, M, z)
        @test isfinite(v_in) && v_in != 0
        @test rsz(4.2 * t200, M, z) == 0
        @test rsz_cyl(4.2 * t200, M, z) != 0
        @test abs(v_in) <= abs(rsz_cyl(2.0 * t200, M, z)) * (1 + 1e-9)
        box = [0.3 -0.3; -0.3 0.3] * Pixell.degree
        shape, wcs = geometry(Pixell.CarClenshawCurtis{Float64}, box, 1.0 * Pixell.arcminute)
        ws = profileworkspace(shape, wcs)
        mr = Enmap(zeros(shape), wcs)
        paint!(mr, ws, rsz, [M], [z], [0.0], [0.0])
        @test count(!iszero, mr.data) > 0 && all(isfinite, mr.data)

        # SZpack rSZ and relativistic kSZ forward the truncation of their interpolators
        kw = (N_logtheta=64, pad=16, N_z=4, N_logM=4, z_min=0.1, z_max=0.6, logM_min=13.0,
              logM_max=15.0, verbose=false)
        ip_y = build_interpolator(tsz; kw...)
        ip_tau = build_interpolator(tau; kw...)
        ip_tau_cyl = build_interpolator(tau_cyl; kw...)
        szp = XGPaint.SZPackRSZProfile(tsz.f_b, tsz.cosmo, 2.6408, ip_y, nothing, 0.01)
        @test XGPaint.truncation(szp) === SphereTruncation(4.0)
        @test XGPaint.compute_θmax(szp, M * M_sun, z) ≈ XGPaint.compute_θmax(tsz, M * M_sun, z) rtol=1e-6
        rk = XGPaint.RKSZpackProfile(ip_y, ip_tau, nothing, nothing; Omega_c=0.261, Omega_b=0.049, h=0.68)
        @test XGPaint.truncation(rk) === SphereTruncation(4.0)
        @test_throws ArgumentError XGPaint.RKSZpackProfile(ip_y, ip_tau_cyl, nothing, nothing)
    end
end
