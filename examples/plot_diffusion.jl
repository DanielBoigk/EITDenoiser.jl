# Figure: truth, LM, posterior mean and std, and individual samples, for one setting of
# landscape_diffusion.jl.   julia --project=. plot_diffusion.jl <results.jls> <out.png> [key index]
using ModularEIT, Ferrite, Serialization, CairoMakie, Statistics
S = deserialize(joinpath(@__DIR__, "landscape_eit.jls"))
R = deserialize(ARGS[1])
keys_ = sort(filter(k -> !(first(k) isa Symbol), collect(keys(R))))
key = keys_[length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1]
X = R[key]
lo, hi = S.σrange
img(x) = lo .+ (hi - lo) .* (clamp.(x, -1, 1) .+ 1) ./ 2
samples = [img(X[:, :, 1, b]) for b in axes(X, 4)]
μ, sd = mean(samples), std(samples)
panels = [("truth", S.truth_img), ("LM (sensitivity damping)", S.lm_img), ("posterior mean", μ)]
append!(panels, [("sample $b", samples[b]) for b in 1:min(4, length(samples))])
fig = Figure(size = (1100, 560))
for (k, (t, im)) in enumerate(panels)
    ax = Axis(fig[(k - 1) ÷ 4 + 1, (k - 1) % 4 + 1]; title = t, aspect = DataAspect(), yreversed = true)
    hidedecorations!(ax)
    heatmap!(ax, permutedims(im); colormap = :grays, colorrange = (lo, hi))
end
ax = Axis(fig[2, 4]; title = "posterior std", aspect = DataAspect(), yreversed = true)
hidedecorations!(ax)
hm = heatmap!(ax, permutedims(sd); colormap = :viridis, colorrange = (0, 0.2))
Colorbar(fig[2, 5], hm)
Label(fig[0, :], "t_start, λ, ζ = $(key)")
save(ARGS[2], fig)
