# Figure of the diffusion showcase: truth, LM, posterior mean and std, samples, resolution map.
#     julia --project=. plot_showcase.jl <results.jls> <out.png> [key index]
using ModularEIT, Ferrite, Serialization, CairoMakie, Statistics
R = deserialize(ARGS[1])
keys_ = sort(filter(k -> k isa Tuple, collect(keys(R))))
key = keys_[length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1]
N = R[:N]
disc = FerriteDiscretization(generate_grid(Quadrilateral, (N, N)))
pp = PixelParametrization(disc, N, N)
pim(θ) = pixel_image(pp, θ)
P = R[key]
lo, hi = 0.2, 1.0
panels = [("truth", pim(R[:truth]), :grays, (lo, hi)), ("Levenberg–Marquardt", pim(R[:θlm]), :grays, (lo, hi)),
          ("posterior mean", pim(mean(P)), :grays, (lo, hi)), ("posterior std", pim(std(P)), :viridis, (0, 0.2)),
          ("sample 1", pim(P[1]), :grays, (lo, hi)), ("sample 2", pim(P[2]), :grays, (lo, hi)),
          ("sample 3", pim(P[3]), :grays, (lo, hi)), ("resolution (confidence)", pim(R[:resolution]), :viridis, (0, 1))]
fig = Figure(size = (1200, 640))
for (k, (t, im, cm, cr)) in enumerate(panels)
    ax = Axis(fig[(k - 1) ÷ 4 + 1, (k - 1) % 4 + 1]; title = t, aspect = DataAspect(), yreversed = true)
    hidedecorations!(ax)
    heatmap!(ax, permutedims(im); colormap = cm, colorrange = cr)
end
Label(fig[0, :], "$(N) × $(N) pixels, $(R[:NEL]) electrodes; t_start, λ, ζ = $(key)")
save(ARGS[2], fig)
