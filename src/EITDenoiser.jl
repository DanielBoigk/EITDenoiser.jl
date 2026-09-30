"""
    EITDenoiser

Diffusion-model priors for electrical impedance tomography: noise-predicting networks (a U-Net
and a local CNN), the variance-preserving SDE, training, and diffusion sampling with data
consistency for reconstructions from ModularEIT.jl.
"""
module EITDenoiser

using Lux, ConcreteStructs, Random, NNlib, Statistics, LinearAlgebra
using Reactant, Enzyme, Optimisers, JLD2, JpegTurbo, ColorTypes
using Lux: Training
using MLDataDevices: CPUDevice

include("models/unet.jl")
include("models/masked_convolution.jl")
include("models/local_score_net.jl")
include("schedule.jl")
include("data.jl")
include("training.jl")
include("sampling.jl")
include("consistency.jl")

export UNet, unet_tinyimagenet64, LocalScoreNet, local_score_net_64
export VPSchedule, alpha_bar, diffuse, denoised_estimate, estimate_noise_level
export load_image_folder, flip_augment, to_model_range, from_model_range
export noisy_batch, train_noise_predictor, save_checkpoint, load_checkpoint, pretrained_unet, scene_unet
export NoisePredictor, diffusion_sample, cpu_device, reactant_device
export LinearizedData, data_prox, parameter_modes, pixel_consistency, polish_sample

end
