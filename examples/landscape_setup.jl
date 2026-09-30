# Shared setup of the landscape showcase: the image as conductivity, the reconstruction mesh
# (N × N squares, boundary layer refined twice), NEL electrodes, data from a finer mesh with 1 %
# noise, and the pixel parametrisation. Environment: N (default 64), NEL (default 32).
using ModularEIT, Ferrite, Random, LinearAlgebra, Printf

N = parse(Int, get(ENV, "N", "64"))
NEL = parse(Int, get(ENV, "NEL", "32"))
bytes = read(pkgdir(ModularEIT, "examples", "data", "1096.u8"))
img = permutedims(reshape(Float64.(bytes) ./ 255, 150, 150))
σrange = (0.2, 1.0)
phantom = image_phantom(img, σrange...)

function boundary_refined(n, levels; band = 2)
    am = AdaptiveMesh(generate_grid(Quadrilateral, (n, n)))
    for _ in 1:levels
        g = current_grid(am)
        near = [i for i in 1:getncells(g) if maximum(maximum(abs.(g.nodes[k].x)) for k in g.cells[i].nodes) > 1 - 2band / n + 1e-12]
        refine_mesh!(am, near)
    end
    return current_grid(am)
end

disc = FerriteDiscretization(boundary_refined(N, 2))
electrodes = angular_electrodes(disc, NEL; coverage = 0.5)
fm = ForwardModel(disc, CompleteElectrodeModel(electrodes, 0.05))
nfine = N >= 128 ? 300 : 150                       # the image is exactly piecewise constant on it
fine = FerriteDiscretization(N >= 128 ? boundary_refined(nfine, 1; band = 2) : generate_grid(Quadrilateral, (nfine, nfine)))
fm_fine = ForwardModel(fine, CompleteElectrodeModel(transfer_electrodes(disc, electrodes, fine), 0.05))
currents = trigonometric_patterns(fm, (NEL - 1) ÷ 2)
noise = RelativeGaussianNoise(0.01)
sim = simulate_data(fine, fm_fine, phantom, currents; noise, rng = MersenneTwister(1096))
pp = PixelParametrization(disc, N, N)
truth = [phantom(c) for c in pp.centres]
@printf "reconstruction mesh %d cells (%d dofs), data mesh %d cells; %d electrodes, %d patterns; %d pixels\n" getncells(disc.grid) ndofs_u(disc) getncells(fine.grid) NEL size(currents, 2) N^2
flush(stdout)

# The least-squares objective in the pixels, and its noise model. With KTRUNC set, the data are
# rotated into their singular patterns relative to the best constant conductivity and truncated
# on both sides: KTRUNC current patterns × KTRUNC measurement modes.
function landscape_objective()
    data = AdjointStateObjective(fm, currents, sim.data)
    K = parse(Int, get(ENV, "KTRUNC", "0"))
    K == 0 && return data, noise, nothing, nothing
    constant = ParametrizedObjective(data, SubspaceParametrization(pp, ones(N^2, 1)))
    c0 = minimize(constant, [0.5], LBFGS(); maxiter = 30).σ[1]    # gradient only: no Jacobian
    p = pattern_svd(disc, fm, currents, sim.data; metric = :L2, noise, reference = fill(c0, ndofs_σ(disc)))
    t = truncate_patterns(p, K; measurements = K)
    @printf "best constant %.4f; %d of %d pairs above 2× noise; kept %d × %d\n" c0 count(p.values .> 2 .* p.noise_levels) length(p.values) K K
    flush(stdout)
    return AdjointStateObjective(fm, t.currents, t.voltages; misfit = ProjectedMisfit(t.projection)), t.noise, c0, t
end
