# Matrix-free (CG) Gauss–Newton on the 128 × 128 / 255-electrode landscape problem: time,
# error and misfit per iteration, and the modelling-error level of the reconstruction model,
# estimated from other scene images (not the test image) simulated on both meshes.
#     N=128 NEL=255 KTRUNC=128 julia --project=. -t 8 benchmark_cg.jl [image folder]
using Serialization, Statistics, JpegTurbo, ColorTypes
include(joinpath(@__DIR__, "landscape_setup.jl"))
BLAS.set_num_threads(Sys.CPU_THREADS)

data, noise_t, c0, t = landscape_objective()
obj = ParametrizedObjective(data, pp)
target = discrepancy_target(data, noise_t)
relerr(θ) = norm(θ - truth) / norm(truth .- mean(truth))
fit(θ) = objective_value(obj, θ) / target

# modelling error: noise-free data of other images from the fine mesh, misfit of their pixel
# values on the reconstruction model
folder = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "Images", "gray")
levels = Float64[]
for name in (get(ENV, "SKIP_LEVELS", "0") == "1" ? () : ("3", "500", "2500", "5000"))
    im = Float64.(gray.(jpeg_decode(Gray, joinpath(folder, name * ".jpg"))))
    ph = image_phantom(im, σrange...)
    Vf = simulate_data(fine, fm_fine, ph, t.currents).clean
    om = ParametrizedObjective(AdjointStateObjective(fm, t.currents, Vf; misfit = ProjectedMisfit(t.projection)), pp)
    push!(levels, objective_value(om, [ph(c) for c in pp.centres]) / target)
end
@printf "modelling error of other scene images: misfit/target %s (the test image: %.2f)\n" string(round.(levels; digits = 2)) fit(truth)
flush(stdout)

t0 = time()
trace = NamedTuple[]
function record(st)
    push!(trace, (; it = st.iteration, time = time() - t0, fit = st.value / target, err = relerr(st.σ)))
    @printf "  it %2d  %6.0f s  misfit/target %8.2f  rel. error %.3f\n" trace[end]...
    flush(stdout)
    return false
end
res = minimize(obj, fill(c0, N^2), GaussNewton(; scaling = :sensitivity, linear_solver = :cg); lower = 0.05,
               maxiter = 40, ftarget = target, callback = record)
@printf "CG Gauss–Newton: %d its, %s, %.0f s, misfit/target %.2f, rel. error %.3f\n" res.iteration res.status time() - t0 fit(res.σ) relerr(res.σ)
serialize(joinpath(@__DIR__, "benchmark_cg_$(N)_$(NEL).jls"), (; trace, levels, θ = res.σ))
