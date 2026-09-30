# Reverse diffusion with an optional data-consistency step (DiffPIR, Zhu et al. 2023):
#     x̂₀ = (x_t - √(1-ᾱ_t) ε̂) / √ᾱ_t              clean-image estimate
#     x̃₀ = consistency(x̂₀, t)                    e.g. a proximal step of the data term
#     ε̃  = (x_t - √ᾱ_t x̃₀) / √(1-ᾱ_t)             the noise that x̃₀ implies
#     x_s = √ᾱ_s x̃₀ + √(1-ᾱ_s) (√(1-ζ) ε̃ + √ζ z)
# ζ = 0 is deterministic (DDIM), ζ = 1 re-draws the noise at every step. Starting at t_start < 1
# from a noised reference image (SDEdit) keeps the coarse structure of the reference.

"""
    NoisePredictor(model, ps, st; device = reactant_device())

A trained network as a function `ε̂(x, t)` of a batch `x` (`H × W × 1 × B`, model range) and a
scalar time `t`. On a Reactant device the forward pass is compiled once per batch shape;
`device = cpu_device()` evaluates the network with plain Lux on the CPU.
"""
struct NoisePredictor{M, P, S, D}
    model::M
    ps::P
    st::S
    device::D
    compiled::Dict{Any, Any}
end
function NoisePredictor(model, ps, st; device = reactant_device(; force = true))
    return NoisePredictor(model, device(ps), device(Lux.testmode(st)), device, Dict{Any, Any}())
end
function (p::NoisePredictor)(x::AbstractArray{<:Real, 4}, t::Real)
    if p.device isa CPUDevice                   # plain Lux on the CPU
        return p.model((Float32.(x), fill(Float32(t), 1, 1, 1, size(x, 4))), p.ps, p.st)[1]
    end
    xd = p.device(Float32.(x))
    td = p.device(fill(Float32(t), 1, 1, 1, size(x, 4)))
    f = get!(p.compiled, size(x)) do
        Reactant.@compile p.model((xd, td), p.ps, p.st)
    end
    return Array(f((xd, td), p.ps, p.st)[1])
end

"""
    diffusion_sample(ε̂, sch, x; steps = 50, ζ = 0, rng = Random.default_rng(), t_start = 1,
                     t_end = 1e-3, reference = nothing, consistency = nothing)

Reverse diffusion from time `t_start` to `t_end` in `steps` steps with the noise predictor
`ε̂(x, t)`, started from `x` (pure noise for `t_start = 1`) or, with a `reference` image, from
the reference noised to `t_start` (SDEdit). `consistency(x̂₀, t)` maps every clean-image estimate
to a data-consistent one (DiffPIR); `ζ ∈ [0, 1]` is the fraction of fresh noise per step.
Returns the final clean-image estimate.
"""
function diffusion_sample(ε̂, s::VPSchedule, x::AbstractArray{<:Real, 4}; steps::Integer = 50, ζ::Real = 0.0,
                          rng::AbstractRNG = Random.default_rng(), t_start::Real = 1.0, t_end::Real = 1.0e-3,
                          reference = nothing, consistency = nothing)
    0 <= ζ <= 1 || throw(ArgumentError("ζ must lie in [0, 1]"))
    x = reference === nothing ? Float32.(x) :
        diffuse(s, Float32.(reference), Float32(t_start), randn(rng, Float32, size(reference)))
    ts = range(Float32(t_start), Float32(t_end); length = steps + 1)
    x0 = x
    for k in 1:steps
        t, tn = ts[k], ts[k + 1]
        a, an = alpha_bar(s, t), alpha_bar(s, tn)
        ε = ε̂(x, t)
        x0 = denoised_estimate(s, x, t, ε)
        if consistency !== nothing
            x0 = consistency(x0, t)
            ε = (x .- sqrt(a) .* x0) ./ sqrt(1 - a)
        end
        noise = ζ > 0 ? sqrt(ζ) .* randn(rng, Float32, size(x)) : zero(x)
        x = sqrt(an) .* x0 .+ sqrt(1 - an) .* (sqrt(1 - ζ) .* ε .+ noise)
    end
    return x0
end
