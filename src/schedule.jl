# Variance-preserving SDE (Song et al. 2021) with a linear β(t) on t ∈ [0, 1]:
#     ᾱ(t) = exp(-βmin t - (βmax - βmin) t² / 2),     x_t = √ᾱ x₀ + √(1-ᾱ) ε.
# The defaults are those the pretrained models were trained with.

"""
    VPSchedule(; βmin = 0.1, βmax = 20)

Noise schedule of the variance-preserving SDE with linear `β(t)` on `t ∈ [0, 1]`.
"""
struct VPSchedule
    βmin::Float32
    βmax::Float32
end
VPSchedule(; βmin = 0.1f0, βmax = 20.0f0) = VPSchedule(Float32(βmin), Float32(βmax))

"""
    alpha_bar(sch, t)

Signal fraction `ᾱ(t) = exp(-βmin t - (βmax - βmin) t² / 2)`.
"""
alpha_bar(s::VPSchedule, t::Real) = exp(-s.βmin * t - (s.βmax - s.βmin) / 2 * t^2)

"""
    diffuse(sch, x0, t, ε)

Noised images `x_t = √ᾱ(t) x₀ + √(1-ᾱ(t)) ε` (`t` a scalar or one value per batch entry).
"""
function diffuse(s::VPSchedule, x0, t, ε)
    a = alpha_bar.(Ref(s), t)
    return sqrt.(a) .* x0 .+ sqrt.(1 .- a) .* ε
end

"""
    denoised_estimate(sch, xt, t, ε)

The estimate `x̂₀ = (x_t - √(1-ᾱ) ε) / √ᾱ` of the clean image from a noise prediction `ε`
(Tweedie's formula for the exact noise predictor).
"""
function denoised_estimate(s::VPSchedule, xt, t, ε)
    a = alpha_bar.(Ref(s), t)
    return (xt .- sqrt.(1 .- a) .* ε) ./ sqrt.(a)
end

"""
    estimate_noise_level(sch, t)

Standard deviation `√(1-ᾱ)/√ᾱ` of the noise in `x_t / √ᾱ`, the scale of the error of the
clean-image estimate at time `t`.
"""
estimate_noise_level(s::VPSchedule, t::Real) = sqrt((1 - alpha_bar(s, t)) / alpha_bar(s, t))
