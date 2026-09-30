# Diffusion part of the showcase: posterior samples of the landscape with a diffusion prior,
# data consistency by the EIT data linearized at the Levenberg–Marquardt reconstruction
# (DiffPIR), started from the noised reconstruction (SDEdit).
#     julia --project=examples examples/landscape_diffusion.jl [model] [device: gpu|cpu]
using EITDenoiser, ModularEIT, Random, LinearAlgebra, Serialization, Printf, Statistics

name = length(ARGS) >= 1 ? ARGS[1] : "scenes_unet"
dev = length(ARGS) >= 2 && ARGS[2] == "cpu" ? cpu_device() : reactant_device(; force = true)
S = deserialize(joinpath(@__DIR__, "landscape_eit.jls"))
S.obj.obj.state = nothing                  # the sparse factorization does not survive serialization
lo, hi = S.σrange
N = S.N
ld = LinearizedData(S.J, S.r, S.θlm; noise = S.η)
model, ps, st = pretrained_unet(name)
ε̂ = NoisePredictor(model, ps, st; device = dev)
sch = VPSchedule()

to_x(img) = Float32.(2 .* (img .- lo) ./ (hi - lo) .- 1)
to_θ(x) = pixel_parameters(S.pp, lo .+ (hi - lo) .* (Float64.(x) .+ 1) ./ 2)
relerr(θ) = norm(θ - S.truth) / norm(S.truth .- mean(S.truth))
misfit(θ) = objective_value(S.obj, θ) / S.target
@printf "LM: rel. error %.3f, misfit/target %.2f\n" relerr(S.θlm) misfit(S.θlm)

B = parse(Int, get(ENV, "B", "8"))
xref = repeat(to_x(S.lm_img), 1, 1, 1, B)
results = Dict{Any, Any}()
for (t_start, λ, ζ) in eval(Meta.parse(get(ENV, "GRID", "[(0.5, 1.0, 0.5)]")))
    dc = pixel_consistency(ld, S.pp; range = S.σrange, λ, schedule = sch)
    t = @elapsed X = diffusion_sample(ε̂, sch, xref; steps = parse(Int, get(ENV, "STEPS", "50")), ζ, t_start,
                                      reference = xref, consistency = dc, rng = Xoshiro(3))
    Θ = [to_θ(X[:, :, 1, b]) for b in 1:B]
    μ = mean(Θ)
    errs, fits = relerr.(Θ), misfit.(Θ)
    @printf "t_start %.2f λ %.1f ζ %.1f: sample err %.3f ± %.3f, mean err %.3f, misfit/target %.2f–%.2f (mean %.2f) [%.0f s]\n" t_start λ ζ mean(errs) std(errs) relerr(μ) minimum(fits) maximum(fits) misfit(μ) t
    # nonlinear polish: a strongly damped correction of every sample to the noise level
    t = @elapsed P = [polish_sample(S.obj, θ; ftarget = S.target, lower = 0.05) for θ in Θ]
    @printf "    polished: sample err %.3f ± %.3f, mean err %.3f, misfit/target %.2f–%.2f, change %.3f [%.0f s]\n" mean(relerr.(P)) std(relerr.(P)) relerr(mean(P)) extrema(misfit.(P))... mean(norm.(P .- Θ) ./ norm.(Θ)) t
    results[(t_start, λ, ζ)] = X
    results[(:polished, t_start, λ, ζ)] = P
end
serialize(joinpath(@__DIR__, "landscape_diffusion_$(name).jls"), results)
