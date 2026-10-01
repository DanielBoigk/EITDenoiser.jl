using EITDenoiser
using Lux, Random, LinearAlgebra, Statistics
using JpegTurbo, ColorTypes
using Test

@testset "EITDenoiser" begin
    rng = Xoshiro(7)

    @testset "VP-SDE schedule" begin
        sch = VPSchedule()                                   # β from 0.1 to 20, as the pretrained models
        @test alpha_bar(sch, 0.0f0) == 1
        ts = range(0, 1; length = 50)
        @test issorted(alpha_bar.(Ref(sch), ts); rev = true)
        @test alpha_bar(sch, 0.3f0) ≈ exp(-0.1 * 0.3 - (20 - 0.1) / 2 * 0.3^2)
        @test alpha_bar(sch, 1.0f0) < 1e-4
        # x_t = √ᾱ x₀ + √(1-ᾱ) ε, and the x₀ estimate inverts it exactly for the true ε
        x0, ε = randn(rng, Float32, 4, 4, 1, 3), randn(rng, Float32, 4, 4, 1, 3)
        xt = diffuse(sch, x0, 0.4f0, ε)
        @test xt ≈ sqrt(alpha_bar(sch, 0.4f0)) .* x0 .+ sqrt(1 - alpha_bar(sch, 0.4f0)) .* ε
        @test denoised_estimate(sch, xt, 0.4f0, ε) ≈ x0 rtol = 1e-4
        # noise level of the x₀ estimate: √(1-ᾱ)/√ᾱ
        @test estimate_noise_level(sch, 0.4f0) ≈ sqrt((1 - alpha_bar(sch, 0.4f0)) / alpha_bar(sch, 0.4f0))
    end

    @testset "image folders" begin
        dir = mktempdir()
        imgs = [rand(rng, Float32, 16, 16) for _ in 1:4]     # (JpegTurbo rejects tiny files)
        for (k, im) in enumerate(imgs)
            write(joinpath(dir, "$(k + 10).jpg"), jpeg_encode(Gray.(im); quality = 100))
        end
        write(joinpath(dir, "notes.txt"), "not an image")
        X, names = load_image_folder(dir; exclude = ["12"])
        @test size(X) == (16, 16, 1, 3)
        @test eltype(X) == Float32 && all(0 .<= X .<= 1)
        @test sort(names) == ["11", "13", "14"]
        k = findfirst(==("13"), names)
        @test X[:, :, 1, k] ≈ imgs[3] atol = 0.03            # JPEG rounding
        # training batches: horizontal flips only (landscapes keep their orientation)
        xb = flip_augment(rng, X)
        for b in 1:3
            @test xb[:, :, 1, b] == X[:, :, 1, b] || xb[:, :, 1, b] == X[:, end:-1:1, 1, b]
        end
        # images in [0, 1] ↔ model range [-1, 1]
        @test to_model_range(X) ≈ 2 .* X .- 1
        @test from_model_range(to_model_range(X)) ≈ X
    end

    @testset "networks: shapes" begin
        unet = UNet((16, 16); channels = [8, 16], block_depth = 1, embedding_dims = 8, use_attention = false)
        ps, st = Lux.setup(rng, unet)
        x, t = randn(rng, Float32, 16, 16, 1, 2), rand(rng, Float32, 1, 1, 1, 2)
        y, _ = unet((x, t), ps, st)
        @test size(y) == size(x)
        net = local_score_net_64(; embedding_dims = 8)
        ps2, st2 = Lux.setup(rng, net)
        y2, _ = net((randn(rng, Float32, 16, 16, 1, 2), rand(rng, Float32, 2)), ps2, st2)
        @test size(y2) == (16, 16, 1, 2)
    end

    @testset "training batches" begin
        sch = VPSchedule()
        X = rand(rng, Float32, 8, 8, 1, 5)
        (xt, t), ε = noisy_batch(rng, sch, X, 4)
        @test size(xt) == (8, 8, 1, 4) && size(t) == (1, 1, 1, 4) && size(ε) == size(xt)
        @test all(0 .< t .<= 1)
        # every x_t is a noised version of some training image (in model range, possibly flipped)
        for b in 1:4
            ab = alpha_bar(sch, t[b])
            x0 = (xt[:, :, 1, b] .- sqrt(1 - ab) .* ε[:, :, 1, b]) ./ sqrt(ab)
            @test any(k -> isapprox(x0, to_model_range(X[:, :, 1, k]); rtol = 1e-3) ||
                           isapprox(x0, to_model_range(X[:, end:-1:1, 1, k]); rtol = 1e-3), 1:5)
        end
    end

    @testset "sampling with an exact denoiser" begin
        sch = VPSchedule()
        xstar = randn(rng, Float32, 6, 6, 1, 1)
        # the exact noise predictor for a single training image x*
        oracle(x, t) = (x .- sqrt(alpha_bar(sch, t)) .* xstar) ./ sqrt(1 - alpha_bar(sch, t))
        for ζ in (0.0, 0.5)
            x = diffusion_sample(oracle, sch, randn(rng, Float32, 6, 6, 1, 1); steps = 30, ζ, rng)
            @test x ≈ xstar rtol = 1e-3
            # SDEdit: start from a noised reference at t_start < 1
            x = diffusion_sample(oracle, sch, zeros(Float32, 6, 6, 1, 1); steps = 20, ζ, rng, t_start = 0.5,
                                 reference = zeros(Float32, 6, 6, 1, 1))
            @test x ≈ xstar rtol = 1e-3
        end
        # data consistency that pins the image to x*: the oracle is not needed any more
        pin(x0, t) = xstar
        x = diffusion_sample((x, t) -> zero(x), sch, randn(rng, Float32, 6, 6, 1, 1); steps = 10, consistency = pin)
        @test x ≈ xstar
    end

    @testset "linearized data consistency" begin
        m, n = 12, 30
        J = randn(rng, m, n) * Diagonal(10.0 .^ range(0, -3; length = n))
        θ0, r0 = randn(rng, n), 0.01 .* randn(rng, m)
        ld = LinearizedData(J, r0, θ0; noise = 0.01)
        θhat = θ0 .+ randn(rng, n)
        for γ in (1e-3, 0.1, 10.0)
            # argmin ‖r0 + J(θ - θ0)‖²/η² + ‖θ - θhat‖²/γ²
            H = J' * J / 0.01^2 + I / γ^2
            θopt = H \ (J' * (J * θ0 - r0) / 0.01^2 + θhat / γ^2)
            @test data_prox(ld, θhat, γ) ≈ θopt rtol = 1e-6
        end
        @test data_prox(ld, θhat, 1e-9) ≈ θhat rtol = 1e-6        # no trust in the data term
        @test ld.s[1:m] ≈ svdvals(J) rtol = 1e-6
        @test abs.(parameter_modes(ld, 4)' * svd(J).V[:, 1:4]) ≈ I atol = 1e-6   # right singular vectors
        # from the Gram matrix directly (e.g. accumulated from row blocks)
        @test data_prox(LinearizedData(J' * J, J' * r0, θ0, :gram; noise = 0.01), θhat, 0.1) ≈ data_prox(ld, θhat, 0.1)
        # batched: columns are independent
        Θ = θ0 .+ randn(rng, n, 3)
        @test data_prox(ld, Θ, 0.1) ≈ reduce(hcat, [data_prox(ld, Θ[:, k], 0.1) for k in 1:3])
        # the linearized misfit of the prox result never exceeds that of θhat
        misfit(θ) = norm(r0 + J * (θ - θ0))
        @test misfit(data_prox(ld, θhat, 0.1)) <= misfit(θhat)
    end
end

using ModularEIT, ModularEITFerrite, Ferrite
@testset "ModularEIT extension" begin
    rng = Xoshiro(11)
    disc = FerriteDiscretization(generate_grid(Quadrilateral, (8, 8)))
    fm = ForwardModel(disc, CompleteElectrodeModel(angular_electrodes(disc, 8; coverage = 0.5), 0.1))
    I_ = trigonometric_patterns(fm, 3)
    σtrue = conductivity(disc, InclusionPhantom(1.0, [CircleInclusion((0.3, 0.2), 0.4, 1.5)]))
    noise = RelativeGaussianNoise(0.01)
    data = AdjointStateObjective(fm, I_, add_noise(forward_neumann(fm, σtrue, I_)[1], noise; rng))
    pp = PixelParametrization(disc, 8, 8)
    obj = ParametrizedObjective(data, pp)
    θ0 = ones(64)
    ld = LinearizedData(obj, θ0; noise)
    js = jacobian_svd(obj, θ0)
    @test ld.s[1:length(js.s)] ≈ js.s atol = 1e-6 * js.s[1]
    @test ld.g0 ≈ js.V * (js.s .* (js.U' * js.r))                   # Jᵀ r
    @test ld.η ≈ sqrt(2 * discrepancy_target(data, noise; τ = 1) / n_residual(data))
    # pixel images in the model range ↔ conductivities in (0.5, 1.5)
    dc = pixel_consistency(ld, pp; range = (0.5, 1.5), λ = 1.0)
    x0 = 0.2f0 .* randn(rng, Float32, 8, 8, 1, 2)
    y = pixel_consistency(ld, pp; range = (0.5, 1.5), clip = false)(x0, 0.5f0)
    @test size(y) == size(x0)
    θx(x) = pixel_parameters(pp, 0.5 .+ (Float64.(x) .+ 1) ./ 2)
    γ = 0.5 * estimate_noise_level(VPSchedule(), 0.5f0)
    @test θx(y[:, :, 1, 2]) ≈ data_prox(ld, θx(x0[:, :, 1, 2]), γ) rtol = 1e-5
    # late times (accurate estimates) change the image less than early ones, which fit the data
    change(t) = norm(dc(x0, t) - x0)
    @test change(1.0f-4) < change(0.1f0) < change(1.0f0)
    @test change(1.0f-4) < 0.05 * norm(x0)
    # without clipping the linearized misfit decreases; with clipping the images stay in range
    lin(x) = norm(js.r + js.U * (js.s .* (js.V' * (θx(x) - ld.θ0))))
    dcn = pixel_consistency(ld, pp; range = (0.5, 1.5), clip = false)
    @test lin(dcn(x0[:, :, :, 1:1], 1.0f0)[:, :, 1, 1]) < lin(x0[:, :, 1, 1])
    @test all(-1 .<= dc(x0, 1.0f0) .<= 1)

    # polishing: a strongly damped nonlinear correction to the noise level, small changes only
    target = discrepancy_target(data, noise)
    θfit = minimize(obj, θ0, GaussNewton(; scaling = :sensitivity); lower = 0.1, ftarget = target).σ
    @test polish_sample(obj, θfit; ftarget = target) == θfit              # already consistent: unchanged
    θoff = θfit .+ 0.05 .* randn(rng, 64)
    objective_value(obj, θoff) > target || (θoff .+= 0.1 .* randn(rng, 64))
    @test objective_value(obj, θoff) > target
    θp = polish_sample(obj, θoff; ftarget = target, lower = 0.1)
    @test objective_value(obj, θp) <= target
    @test norm(θp - θoff) < norm(θoff - θfit)                             # a nudge, not a new fit
    @test all(θp .>= 0.1)
end
