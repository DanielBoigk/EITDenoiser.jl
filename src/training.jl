# Training / fine-tuning of noise predictors ε̂(x_t, t) with the denoising score matching loss
# ‖ε̂(x_t, t) - ε‖² on the GPU through Reactant, and checkpoints.

"""
    noisy_batch(rng, sch, X, batchsize; tmin = 1e-4)

A training batch `((x_t, t), ε)` from the images `X` (`H × W × 1 × N` in `[0, 1]`): random
images, mirrored with probability 1/2, mapped to `[-1, 1]` and noised to uniformly random times
`t ∈ [tmin, 1]` (`t` of size `1 × 1 × 1 × batchsize`, as the U-Net expects).
"""
function noisy_batch(rng::AbstractRNG, s::VPSchedule, X::AbstractArray{<:Real, 4}, batchsize::Integer;
                     tmin::Real = 1.0f-4)
    idx = rand(rng, axes(X, 4), batchsize)
    x0 = to_model_range(flip_augment(rng, Float32.(X[:, :, :, idx])))
    t = Float32(tmin) .+ (1 - Float32(tmin)) .* rand(rng, Float32, 1, 1, 1, batchsize)
    ε = randn(rng, Float32, size(x0))
    return (diffuse(s, x0, t, ε), t), ε
end

"""
    train_noise_predictor(model, ps, st, X; steps, batchsize = 128, lr = 1e-4, schedule = VPSchedule(),
                          rng = Random.default_rng(), device = reactant_device(), callback = nothing)

Trains (or fine-tunes, from the parameters `ps`, `st`) the noise predictor `model((x_t, t))` on
the images `X` with the loss `‖ε̂ - ε‖²` (NAdam with gradient clipping). `callback(step, loss,
train_state)` is called after every step. Returns the CPU parameters, states and the losses.
"""
function train_noise_predictor(model, ps, st, X::AbstractArray{<:Real, 4}; steps::Integer, batchsize::Integer = 128,
                               lr::Real = 1.0f-4, schedule::VPSchedule = VPSchedule(),
                               rng::AbstractRNG = Random.default_rng(), device = reactant_device(; force = true),
                               callback = nothing)
    cdev = cpu_device()
    opt = Optimisers.OptimiserChain(Optimisers.ClipNorm(1.0f0), Optimisers.NAdam(Float32(lr)))
    ts = Training.TrainState(model, device(ps), device(st), opt)
    losses = Float32[]
    for step in 1:steps
        (xt, t), ε = noisy_batch(rng, schedule, X, batchsize)
        _, loss, _, ts = Training.single_train_step!(AutoEnzyme(), MSELoss(), ((device(xt), device(t)), device(ε)), ts;
                                                     return_gradients = Val(false))
        push!(losses, Float32(loss))
        callback === nothing || callback(step, losses[end], ts)
    end
    return cdev(ts.parameters), cdev(ts.states), losses
end

"""
    save_checkpoint(dir, ps, st)
    load_checkpoint(dir)

Parameters and states as `ps_latestvn.jld2` / `st_latestvn.jld2` in `dir` (the format of the
original training scripts; variables `ps_cpu`, `st_cpu`).
"""
function save_checkpoint(dir::AbstractString, ps, st)
    mkpath(dir)
    JLD2.jldsave(joinpath(dir, "ps_latestvn.jld2"); ps_cpu = ps)
    JLD2.jldsave(joinpath(dir, "st_latestvn.jld2"); st_cpu = st)
    return dir
end
load_checkpoint(dir::AbstractString) =
    (JLD2.load(joinpath(dir, "ps_latestvn.jld2"), "ps_cpu"), JLD2.load(joinpath(dir, "st_latestvn.jld2"), "st_cpu"))

"""
    pretrained_unet(name = "scenes_unet")

A pretrained 64 × 64 U-Net from `pretrained/<name>`: `(model, ps, st)` on the CPU.
`"tinyimagenet_unet"`: trained on TinyImageNet with rotations and flips; `"scenes_unet"`:
fine-tuned from it on natural scenes with horizontal flips only.
"""
function pretrained_unet(name::AbstractString = "scenes_unet")
    ps, st = load_checkpoint(joinpath(pkgdir(@__MODULE__), "pretrained", name))
    return unet_tinyimagenet64(; embedding_dims = 32), ps, st
end
