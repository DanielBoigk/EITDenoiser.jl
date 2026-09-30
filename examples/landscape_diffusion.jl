# Diffusion part of the showcase: posterior samples of the landscape with a diffusion prior,
# data consistency by the EIT data linearized at the Levenberg–Marquardt reconstruction
# (DiffPIR), started from the noised reconstruction (SDEdit), then polished with the nonlinear
# model where needed. Needs landscape_eit.jl first (same N, NEL, KTRUNC).
#     N=128 NEL=255 KTRUNC=128 julia --project=. -t 8 landscape_diffusion.jl [model]
using EITDenoiser, Serialization, Statistics
include(joinpath(@__DIR__, "landscape_setup.jl"))
BLAS.set_num_threads(Sys.CPU_THREADS)

name = length(ARGS) >= 1 ? ARGS[1] : (N == 128 ? "scenes_unet_128" : "scenes_unet")
data, noise_t, _, _ = landscape_objective()
obj = ParametrizedObjective(data, pp)
aem = get(ENV, "AEM", "0") == "1"
if aem                                   # modelling error accounted for (landscape_aem.jl)
    A = deserialize(joinpath(@__DIR__, "landscape_aem_$(N)_$(NEL).jls"))
    obj = ApproximationErrorObjective(obj, A.ae)
    t_gram = @elapsed ld = LinearizedData(A.G, A.g, A.θ, :gram; noise = 1.0)   # whitened: unit noise
    S = (; θlm = A.θ, target = A.atarget)
    A = nothing
    m_res = n_residual(obj)
else
    S = deserialize(joinpath(@__DIR__, "landscape_eit_$(N)_$(NEL).jls"))
    m_res = size(S.J, 1)
    t_gram = @elapsed ld = LinearizedData(S.J, S.r, S.θlm; noise = S.η)  # Gram matrix JᵀJ, eigendecomposition
    S = (; S.θlm, S.target)                                        # release the Jacobian
end
GC.gc()
lo, hi = σrange
@printf "Gram eigendecomposition for %d residuals × %d pixels: %.0f s\n" m_res N^2 t_gram
flush(stdout)
k = count(>=(1e-2 * ld.s[1]), ld.s)                     # resolution map: truncation at rtol = 1e-2
R = vec(sum(abs2, parameter_modes(ld, k); dims = 2))
model, ps, st = pretrained_unet(name)
ε̂ = NoisePredictor(model, ps, st)
sch = VPSchedule()

to_x(img) = Float32.(2 .* (img .- lo) ./ (hi - lo) .- 1)
to_θ(x) = pixel_parameters(pp, lo .+ (hi - lo) .* (clamp.(Float64.(x), -1, 1) .+ 1) ./ 2)
relerr(θ) = norm(θ - truth) / norm(truth .- mean(truth))
misfit(θ) = objective_value(obj, θ) / S.target
@printf "LM: rel. error %.3f, misfit/target %.2f\n" relerr(S.θlm) misfit(S.θlm)
flush(stdout)

B = parse(Int, get(ENV, "B", "8"))
xref = repeat(to_x(pixel_image(pp, S.θlm)), 1, 1, 1, B)
results = Dict{Any, Any}(:truth => truth, :θlm => S.θlm, :resolution => R, :N => N, :NEL => NEL)
for (t_start, λ, ζ) in eval(Meta.parse(get(ENV, "GRID", "[(0.5, 1.0, 0.5)]")))
    dc = pixel_consistency(ld, pp; range = σrange, λ, schedule = sch)
    t = @elapsed X = diffusion_sample(ε̂, sch, xref; steps = parse(Int, get(ENV, "STEPS", "50")), ζ, t_start,
                                      reference = xref, consistency = dc, rng = Xoshiro(3))
    Θ = [to_θ(X[:, :, 1, b]) for b in 1:B]
    errs, fits = relerr.(Θ), misfit.(Θ)
    @printf "t_start %.2f λ %.1f ζ %.1f: sample err %.3f ± %.3f, mean err %.3f, misfit/target %.2f–%.2f (mean %.2f) [%.0f s]\n" t_start λ ζ mean(errs) std(errs) relerr(mean(Θ)) extrema(fits)... misfit(mean(Θ)) t
    flush(stdout)
    if get(ENV, "POLISH", "1") == "0"
        results[(t_start, λ, ζ)] = Θ
        continue
    end
    t = @elapsed P = [polish_sample(obj, θ; ftarget = S.target, lower = 0.05, linear_solver = N >= 128 ? :cg : :auto)
                      for θ in Θ]
    @printf "    polished: sample err %.3f ± %.3f, mean err %.3f, misfit/target %.2f–%.2f, change %.3f [%.0f s]\n" mean(relerr.(P)) std(relerr.(P)) relerr(mean(P)) extrema(misfit.(P))... mean(norm.(P .- Θ) ./ norm.(Θ)) t
    flush(stdout)
    results[(t_start, λ, ζ)] = P
end
serialize(joinpath(@__DIR__, "landscape_diffusion_$(N)_$(NEL)_$(name)$(aem ? "_aem" : "").jls"), results)
