# Fine-tune the 64 × 64 U-Net (pretrained on TinyImageNet) on the natural-scene images
# (Intel scene classification data, grayscale, 64 × 64), horizontal flips only. Image 1096,
# the EIT test image, is excluded; 200 further images are held out for validation.
#
#     julia --project scripts/train_scenes.jl [steps] [image folder]
using EITDenoiser, Lux, Reactant, Random, Statistics, Printf, Dates

steps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20_000
folder = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "..", "..", "Images", "64")
out = joinpath(pkgdir(EITDenoiser), "pretrained", "scenes_unet")

X, names = load_image_folder(folder; exclude = ["1096"])
rng = Xoshiro(2026)
perm = randperm(rng, size(X, 4))
val, train = perm[1:200], perm[201:end]
Xtrain, Xval = X[:, :, :, train], X[:, :, :, val]
@printf "%d training, %d validation images of size %s\n" size(Xtrain, 4) size(Xval, 4) string(size(X)[1:2])

model, ps, st = pretrained_unet("tinyimagenet_unet")
sch = VPSchedule()
dev = reactant_device(; force = true)

# validation loss: fixed noise and times, test-mode network
vrng = Xoshiro(1)
(vx, vt), vε = noisy_batch(vrng, sch, Xval, 200)
function validation_loss(ps, st)
    pred = let xd = dev(vx), td = dev(vt), psd = dev(ps), std = dev(Lux.testmode(st))
        g = Reactant.@compile model((xd, td), psd, std)
        Array(g((xd, td), psd, std)[1])
    end
    return mean(abs2, pred .- vε)
end
@printf "validation loss before: %.5f\n" validation_loss(ps, st)

t0 = now()
window = Float32[]
function cb(step, loss, ts)
    push!(window, loss)
    if step % 500 == 0
        @printf "step %6d  train loss %.5f  (%s)\n" step mean(window) string(round(now() - t0, Second))
        empty!(window)
        flush(stdout)
    end
    if step % 5000 == 0
        save_checkpoint(out, cpu_device()(ts.parameters), cpu_device()(ts.states))
    end
end
ps, st, losses = train_noise_predictor(model, ps, st, Xtrain; steps, batchsize = 128, lr = 1.0f-4, schedule = sch,
                                       rng, device = dev, callback = cb)
save_checkpoint(out, ps, st)
@printf "validation loss after: %.5f\n" validation_loss(ps, st)
