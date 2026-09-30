# EIT part of the diffusion showcase: data, sensitivity-damped Levenberg–Marquardt
# reconstruction on 64 × 64 pixels, and the data linearized there. Saves everything to
# landscape_eit.jls for landscape_diffusion.jl.
using ModularEIT, Ferrite, Random, LinearAlgebra, Serialization, Printf

bytes = read(pkgdir(ModularEIT, "examples", "data", "1096.u8"))
img = permutedims(reshape(Float64.(bytes) ./ 255, 150, 150))
σmin, σmax = 0.2, 1.0
phantom = image_phantom(img, σmin, σmax)

N = 64
am = AdaptiveMesh(generate_grid(Quadrilateral, (N, N)))
for _ in 1:2
    g = current_grid(am)
    near = [i for i in 1:getncells(g) if maximum(maximum(abs.(g.nodes[n].x)) for n in g.cells[i].nodes) > 1 - 2 / N * 2 + 1e-12]
    refine_mesh!(am, near)
end
disc = FerriteDiscretization(current_grid(am))
electrodes = angular_electrodes(disc, 32; coverage = 0.5)
fm = ForwardModel(disc, CompleteElectrodeModel(electrodes, 0.05))
fine = FerriteDiscretization(generate_grid(Quadrilateral, (150, 150)))
fm_fine = ForwardModel(fine, CompleteElectrodeModel(transfer_electrodes(disc, electrodes, fine), 0.05))
currents = trigonometric_patterns(fm, 15)
noise = RelativeGaussianNoise(0.01)
sim = simulate_data(fine, fm_fine, phantom, currents; noise, rng = MersenneTwister(1096))
pp = PixelParametrization(disc, N, N)
truth = [phantom(c) for c in pp.centres]
data = AdjointStateObjective(fm, currents, sim.data)
obj = ParametrizedObjective(data, pp)
target = discrepancy_target(data, noise)
@printf "%d cells, %d pixels, %d residuals\n" getncells(disc.grid) parameter_count(pp) n_residual(obj)

relerr(θ) = norm(θ - truth) / norm(truth .- sum(truth) / length(truth))
c0 = minimize(ParametrizedObjective(data, SubspaceParametrization(pp, ones(N * N, 1))), [0.5], GaussNewton()).σ[1]
t = @elapsed lm = minimize(obj, fill(c0, N * N), GaussNewton(; scaling = :sensitivity); lower = 0.05, maxiter = 40, ftarget = target)
@printf "LM: %d its, %s, misfit/target %.2f, rel. error %.3f (%.0f s)\n" lm.iteration lm.status objective_value(obj, lm.σ) / target relerr(lm.σ) t
r = zeros(n_residual(obj)); J = zeros(length(r), N * N)
residual_and_jacobian!(r, J, obj, lm.σ)
η = ModularEIT._residual_noise_std(obj, noise)
serialize(joinpath(@__DIR__, "landscape_eit.jls"),
          (; truth, θlm = lm.σ, J, r, η, target, σrange = (σmin, σmax), N, pp, obj,
           truth_img = pixel_image(pp, truth), lm_img = pixel_image(pp, lm.σ)))
