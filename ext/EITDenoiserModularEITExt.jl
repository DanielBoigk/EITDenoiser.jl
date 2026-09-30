module EITDenoiserModularEITExt

using EITDenoiser, ModularEIT

"""
    LinearizedData(obj, θ0; noise)

The residual of the ModularEIT least-squares objective `obj` linearized at `θ0` (one Jacobian
evaluation). `noise` is the residual noise level `η`, or a ModularEIT noise model converted to
it (as for `posterior_std`).
"""
function EITDenoiser.LinearizedData(obj::ModularEIT.AbstractObjective, θ0::AbstractVector; noise)
    r = zeros(n_residual(obj))
    J = zeros(length(r), length(θ0))
    residual_and_jacobian!(r, J, obj, Vector{Float64}(θ0))
    return LinearizedData(J, r, θ0; noise = ModularEIT._residual_noise_std(obj, noise))
end

function EITDenoiser.pixel_consistency(ld::LinearizedData, pixels::ModularEIT.PixelParametrization; range,
                                       λ::Real = 1.0, schedule::VPSchedule = VPSchedule(), clip::Bool = true)
    lo, hi = Float64.(Tuple(range))
    n, m = pixels.n, pixels.m
    return function (x0, t)
        size(x0)[1:3] == (n, m, 1) || throw(DimensionMismatch("images must be $n × $m × 1 × B"))
        B = size(x0, 4)
        Θ = reduce(hcat, [pixel_parameters(pixels, lo .+ (hi - lo) .* (Float64.(x0[:, :, 1, b]) .+ 1) ./ 2) for b in 1:B])
        γ = (hi - lo) / 2 * estimate_noise_level(schedule, t) / sqrt(λ)
        Θ = data_prox(ld, Θ, γ)
        out = similar(x0)
        for b in 1:B
            img = 2 .* (pixel_image(pixels, Θ[:, b]) .- lo) ./ (hi - lo) .- 1
            out[:, :, 1, b] .= clip ? clamp.(img, -1, 1) : img
        end
        return out
    end
end

function EITDenoiser.polish_sample(obj::ModularEIT.AbstractObjective, θ::AbstractVector; ftarget::Real,
                                   λ::Real = 1.0, maxiter::Integer = 20, lower = nothing, upper = nothing,
                                   linear_solver::Symbol = :auto)
    objective_value(obj, θ) <= ftarget && return Vector{Float64}(θ)   # (no Jacobian needed)
    res = minimize(obj, θ, GaussNewton(; scaling = :sensitivity, λ, linear_solver); lower, upper, maxiter, ftarget)
    return res.σ
end

end
