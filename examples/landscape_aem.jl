# Approximation-error model for the landscape showcase: the modelling error of the reconstruction
# model (coarser mesh, pixels) estimated from scene images simulated on both meshes, then
# matrix-free Levenberg–Marquardt on the corrected misfit, and the Gram matrix of its Jacobian
# for the diffusion step (landscape_diffusion.jl with AEM=1).
#     N=128 NEL=255 KTRUNC=128 NSAMPLES=200 julia --project=. -t 8 landscape_aem.jl [image folder]
# FTARGET: stop at this multiple of the discrepancy target (default 1, 0: never), MAXITER (default
# 40), STOP and KAPPA (below); TAG is appended to the output name (landscape_diffusion.jl with
# AEM=1 and the same TAG). GRAM=0 skips the Gram matrix and the output.
using Serialization, Statistics, Random, JpegTurbo, ColorTypes
include(joinpath(@__DIR__, "landscape_setup.jl"))
BLAS.set_num_threads(Sys.CPU_THREADS)

data, noise_t, c0, t = landscape_objective()
obj = ParametrizedObjective(data, pp)
target = discrepancy_target(data, noise_t)
relerr(θ) = norm(θ - truth) / norm(truth .- mean(truth))

# modelling-error samples: scene images (not the test image), noise-free fine data, residual of
# the reconstruction model at the pixel values
folder = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "Images", "gray")
names_ = filter(n -> endswith(n, ".jpg") && n != "1096.jpg", readdir(folder))
K = parse(Int, get(ENV, "NSAMPLES", "200"))
chosen = names_[randperm(Xoshiro(7), length(names_))[1:K]]
R = zeros(n_residual(obj), K)
ts = @elapsed for (k, name) in enumerate(chosen)
    im = Float64.(gray.(jpeg_decode(Gray, joinpath(folder, name))))
    ph = image_phantom(im, σrange...)
    Vf = simulate_data(fine, fm_fine, ph, t.currents).clean
    om = ParametrizedObjective(AdjointStateObjective(fm, t.currents, Vf; misfit = ProjectedMisfit(t.projection)), pp)
    residual!(view(R, :, k), om, [ph(c) for c in pp.centres])
    k % 25 == 0 && (@printf "  %d samples\n" k; flush(stdout))
end
η = ModularEIT._residual_noise_std(data, noise_t)
ta = @elapsed ae = ApproximationError(R; noise = η)
aobj = ApproximationErrorObjective(obj, ae)
atarget = discrepancy_target(aobj)
@printf "%d samples in %.0f s; model in %.0f s: rank %d, noise %.3g, floor ν %.3g (%.1f× the noise)\n" K ts ta length(ae.w) η ae.ν ae.ν / η
@printf "misfit/target of the true image: plain %.2f, with the modelling error %.2f\n" objective_value(obj, truth) / target objective_value(aobj, truth) / atarget
flush(stdout)

# Stopping: STOP=target at FTARGET × the discrepancy target; STOP=significance (default) when a
# step lowers the whitened misfit by less than KAPPA (default 3) standard deviations √(m/2) of
# its noise-only value (χ²_m / 2), returning the iterate before that step: LM with the modelling
# error is semi-convergent, and once the misfit stalls the steps fit noise.
significance = get(ENV, "STOP", "significance") == "significance"
κ = parse(Float64, get(ENV, "KAPPA", "3"))
δJ = κ * sqrt(n_residual(aobj) / 2)
prev = Ref{Any}(nothing)
t0 = time()
function record(st)
    @printf "  it %2d  %6.0f s  misfit/target %8.4f  rel. error %.3f\n" st.iteration time() - t0 st.value / atarget relerr(st.σ)
    flush(stdout)
    stall = significance && prev[] !== nothing && prev[].value - st.value < δJ
    stall || (prev[] = (; value = st.value, σ = copy(st.σ), iteration = st.iteration))
    return stall
end
τ = parse(Float64, get(ENV, "FTARGET", significance ? "0" : "1"))
res = minimize(aobj, fill(c0, N^2), GaussNewton(; scaling = :sensitivity, linear_solver = :cg); lower = 0.05,
               maxiter = parse(Int, get(ENV, "MAXITER", "40")), ftarget = τ * atarget, callback = record)
θ, its = res.status === :callback ? (prev[].σ, prev[].iteration) : (res.σ, res.iteration)
@printf "LM with the modelling error: %d its (%s), returned iterate %d, %.0f s, rel. error %.3f\n" res.iteration res.status its time() - t0 relerr(θ)
flush(stdout)
get(ENV, "GRAM", "1") == "0" && exit()
tg = @elapsed G, g = jacobian_gram(aobj, θ)
@printf "Gram matrix: %.0f s\n" tg
serialize(joinpath(@__DIR__, "landscape_aem_$(N)_$(NEL)$(get(ENV, "TAG", "")).jls"), (; θ, G, g, ae, atarget, chosen))
