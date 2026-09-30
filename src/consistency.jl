# Data consistency with a linearized forward model r(θ) ≈ r₀ + J (θ - θ₀), e.g. EIT linearized
# at a Levenberg–Marquardt reconstruction θ₀. The proximal step
#     argmin_θ ‖r₀ + J (θ - θ₀)‖² / η² + ‖θ - θ̂‖² / γ²
# is, with the SVD J = U S Vᵀ and b = r₀ + J (θ̂ - θ₀),
#     θ = θ̂ - V diag(sᵢ / (sᵢ² + η²/γ²)) Uᵀ b:
# directions the data determine (sᵢ ≫ η/γ) are taken from the data, the others stay at θ̂.

"""
    LinearizedData(J, r0, θ0; noise)

Linearized residual `r(θ) ≈ r0 + J (θ - θ0)` with independent Gaussian noise of standard
deviation `noise` per residual entry, for [`data_prox`](@ref). With ModularEIT loaded,
`LinearizedData(obj, θ0; noise)` builds it from a least-squares objective.
"""
struct LinearizedData
    J::Matrix{Float64}
    U::Matrix{Float64}
    s::Vector{Float64}
    V::Matrix{Float64}
    r0::Vector{Float64}
    θ0::Vector{Float64}
    η::Float64
end
function LinearizedData(J::AbstractMatrix, r0::AbstractVector, θ0::AbstractVector; noise::Real)
    size(J) == (length(r0), length(θ0)) || throw(DimensionMismatch("J must be $(length(r0)) × $(length(θ0))"))
    noise > 0 || throw(ArgumentError("the noise level must be positive"))
    F = svd(Matrix{Float64}(J))
    return LinearizedData(Matrix{Float64}(J), F.U, F.S, F.V, Vector{Float64}(r0), Vector{Float64}(θ0), Float64(noise))
end

"""
    data_prox(ld, θhat, γ)

`argmin_θ ‖r0 + J (θ - θ0)‖² / η² + ‖θ - θhat‖² / γ²` for the [`LinearizedData`](@ref) `ld`:
the data-consistent parameters closest to `θhat`, where `γ` is the trust in `θhat` (its
expected error). `θhat` may be a matrix with one parameter vector per column.
"""
function data_prox(ld::LinearizedData, θhat::AbstractVecOrMat, γ::Real)
    b = ld.r0 .+ ld.J * (θhat .- ld.θ0)
    f = ld.s ./ (ld.s .^ 2 .+ (ld.η / γ)^2)
    return θhat .- ld.V * (f .* (ld.U' * b))
end

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
    polish_sample(obj, θ; ftarget, λ = 1, maxiter = 20, lower = nothing, upper = nothing)

Nonlinear correction of a sample `θ` (e.g. from [`diffusion_sample`](@ref) with linearized data
consistency) until the misfit of the least-squares objective `obj` reaches `ftarget` (the
discrepancy target): Levenberg–Marquardt with sensitivity damping and a *large* initial
damping `λ` (relative), so that the correction is a small nudge along the data-determined
directions. Weak damping would re-fit the data and destroy the texture of the sample. Samples
that already fit are returned unchanged. Defined in the ModularEIT extension.
"""
function polish_sample end
