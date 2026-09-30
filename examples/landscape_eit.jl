# EIT part of the diffusion showcase: sensitivity-damped Levenberg–Marquardt on N × N pixels and
# the data linearized at the reconstruction, saved for landscape_diffusion.jl.
#     N=64 NEL=32 julia --project=. landscape_eit.jl
#     N=128 NEL=255 KTRUNC=128 julia --project=. -t 8 landscape_eit.jl
using Serialization, Statistics
include(joinpath(@__DIR__, "landscape_setup.jl"))
BLAS.set_num_threads(Sys.CPU_THREADS)

data, noise_t, c0 = landscape_objective()
obj = ParametrizedObjective(data, pp)
target = discrepancy_target(data, noise_t)
relerr(θ) = norm(θ - truth) / norm(truth .- mean(truth))
c0 === nothing && (c0 = minimize(ParametrizedObjective(data, SubspaceParametrization(pp, ones(N^2, 1))), [0.5], LBFGS()).σ[1])
@printf "%d residuals, %d pixels\n" n_residual(obj) N^2
t = @elapsed lm = minimize(obj, fill(c0, N^2), GaussNewton(; scaling = :sensitivity); lower = 0.05, maxiter = 40,
                           ftarget = target, verbose = true)
@printf "LM: %d its, %s, misfit/target %.2f, rel. error %.3f (%.0f s)\n" lm.iteration lm.status objective_value(obj, lm.σ) / target relerr(lm.σ) t
r = zeros(n_residual(obj)); J = zeros(length(r), N^2)
t = @elapsed residual_and_jacobian!(r, J, obj, lm.σ)
η = ModularEIT._residual_noise_std(data, noise_t)
serialize(joinpath(@__DIR__, "landscape_eit_$(N)_$(NEL).jls"), (; θlm = lm.σ, J, r, η, target, lm_history = lm.history))
@printf "linearized (%.0f s), saved\n" t
