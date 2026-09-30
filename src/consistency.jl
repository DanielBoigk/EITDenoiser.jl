# Data consistency with a linearized forward model r(θ) ≈ r₀ + J (θ - θ₀), e.g. EIT linearized
# at a Levenberg–Marquardt reconstruction θ₀. The proximal step
#     argmin_θ ‖r₀ + J (θ - θ₀)‖² / η² + ‖θ - θ̂‖² / γ²
# is, with λ = η²/γ²,
#     θ = θ̂ - (JᵀJ + λI)⁻¹ Jᵀ (r₀ + J (θ̂ - θ₀)):
# directions the data determine (sᵢ² ≫ λ) are taken from the data, the others stay at θ̂.
# Only the Gram matrix G = JᵀJ = V S² Vᵀ (n × n) and g₀ = Jᵀr₀ enter, since
# Jᵀ(r₀ + J d) = g₀ + G d:
#     θ = θ̂ - V diag(1/(sᵢ² + λ)) Vᵀ g₀ - V diag(sᵢ²/(sᵢ² + λ)) Vᵀ (θ̂ - θ₀).
# The size is independent of the number of residuals, and G can be accumulated from row blocks
# of J without storing J.

"""
    LinearizedData(J, r0, θ0; noise)
    LinearizedData(G, g0, θ0, :gram; noise)

Linearized residual `r(θ) ≈ r0 + J (θ - θ0)` with independent Gaussian noise of standard
deviation `noise` per residual entry, for [`data_prox`](@ref), stored through the Gram matrix
`G = JᵀJ` (its eigenvectors `V` and the singular values `s` of `J`, decreasing) and
`g0 = Jᵀ r0`; `J` itself is not kept. The second form takes `G` and `g0` directly, e.g.
accumulated from row blocks of a Jacobian too large to store. With ModularEIT loaded,
`LinearizedData(obj, θ0; noise)` builds it from a least-squares objective.
"""
struct LinearizedData
    V::Matrix{Float64}
    s::Vector{Float64}
    g0::Vector{Float64}
    θ0::Vector{Float64}
    η::Float64
end
function LinearizedData(J::AbstractMatrix, r0::AbstractVector, θ0::AbstractVector; noise::Real)
    size(J) == (length(r0), length(θ0)) || throw(DimensionMismatch("J must be $(length(r0)) × $(length(θ0))"))
    return _from_gram!(Matrix{Float64}(J' * J), J' * r0, θ0, noise)     # (the Gram matrix is ours)
end
LinearizedData(G::AbstractMatrix, g0::AbstractVector, θ0::AbstractVector, ::Val{:gram}; noise::Real) =
    _from_gram!(Matrix{Float64}(G), g0, θ0, noise)                      # (copy: G is the caller's)

# eigendecomposition of G, overwriting it
function _from_gram!(G::Matrix{Float64}, g0::AbstractVector, θ0::AbstractVector, noise::Real)
    size(G) == (length(θ0), length(θ0)) == (length(g0), length(g0)) ||
        throw(DimensionMismatch("G must be $(length(θ0)) × $(length(θ0)), g0 of length $(length(θ0))"))
    noise > 0 || throw(ArgumentError("the noise level must be positive"))
    E = eigen!(Symmetric(G))
    p = sortperm(E.values; rev = true)
    return LinearizedData(E.vectors[:, p], sqrt.(max.(E.values[p], 0.0)), Vector{Float64}(g0),
                          Vector{Float64}(θ0), Float64(noise))
end
LinearizedData(G::AbstractMatrix, g0::AbstractVector, θ0::AbstractVector, s::Symbol; noise::Real) =
    s === :gram ? LinearizedData(G, g0, θ0, Val(:gram); noise) : throw(ArgumentError("unknown form :$s"))

"""
    data_prox(ld, θhat, γ)

`argmin_θ ‖r0 + J (θ - θ0)‖² / η² + ‖θ - θhat‖² / γ²` for the [`LinearizedData`](@ref) `ld`:
the data-consistent parameters closest to `θhat`, where `γ` is the trust in `θhat` (its
expected error). `θhat` may be a matrix with one parameter vector per column.
"""
function data_prox(ld::LinearizedData, θhat::AbstractVecOrMat, γ::Real)
    λ = (ld.η / γ)^2
    s2 = ld.s .^ 2
    c = (ld.V' * ld.g0) ./ (s2 .+ λ) .+ (s2 ./ (s2 .+ λ)) .* (ld.V' * (θhat .- ld.θ0))
    return θhat .- ld.V * c
end

"""
    parameter_modes(ld, k)

The `k` leading parameter modes `v₁, …, v_k` of the [`LinearizedData`](@ref) `ld` (right
singular vectors of `J`, orthonormal), e.g. for the resolution map `Σᵢ fᵢ vᵢ²`.
"""
parameter_modes(ld::LinearizedData, k::Integer) = ld.V[:, 1:k]

"""
    pixel_consistency(ld, pixels; range, λ = 1, schedule = VPSchedule(), clip = true)

The data-consistency map `(x̂₀, t) -> x̃₀` for [`diffusion_sample`](@ref) when the parameters of
`ld` are the pixels of a ModularEIT `PixelParametrization` `pixels`: images in the model range
are mapped to conductivities in `range = (σmin, σmax)`, moved by [`data_prox`](@ref) with the
trust `γ = (σmax - σmin)/2 · estimate_noise_level(t) / √λ` (DiffPIR's weighting), and mapped
back, clipped to the model range `[-1, 1]` (`clip`; the linearized data term knows no bounds, and
early, weakly trusted estimates would otherwise leave the range of the training images).
Defined in the ModularEIT extension.
"""
function pixel_consistency end

"""
    polish_sample(obj, θ; ftarget, λ = 1, maxiter = 20, lower = nothing, upper = nothing,
                  linear_solver = :auto)

Nonlinear correction of a sample `θ` (e.g. from [`diffusion_sample`](@ref) with linearized data
consistency) until the misfit of the least-squares objective `obj` reaches `ftarget` (the
discrepancy target): Levenberg–Marquardt with sensitivity damping and a *large* initial
damping `λ` (relative), so that the correction is a small nudge along the data-determined
directions. Weak damping would re-fit the data and destroy the texture of the sample. Samples
that already fit are returned unchanged. `linear_solver = :cg` uses ModularEIT's matrix-free
Gauss–Newton (for large problems). Defined in the ModularEIT extension.
"""
function polish_sample end
