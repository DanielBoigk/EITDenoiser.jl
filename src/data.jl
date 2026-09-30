# Image folders (grayscale JPEGs of one size) as training data, and the conversions between
# images in [0, 1] and the model range [-1, 1].

"""
    load_image_folder(dir; exclude = String[], limit = typemax(Int))

All `.jpg`/`.jpeg` images of `dir` as grayscale `Float32` values in `[0, 1]`, an array
`H × W × 1 × N` (row 1 at the top), and their names (file names without extension). Images whose
name is in `exclude` are skipped, e.g. a test image that must not be part of the training set.
All images must have the same size.
"""
function load_image_folder(dir::AbstractString; exclude = String[], limit::Integer = typemax(Int))
    files = sort(filter(f -> lowercase(splitext(f)[2]) in (".jpg", ".jpeg"), readdir(dir)))
    ex = Set(string.(exclude))
    names = String[]
    imgs = Matrix{Float32}[]
    for f in files
        name = splitext(f)[1]
        name in ex && continue
        length(imgs) >= limit && break
        img = Float32.(gray.(jpeg_decode(Gray, joinpath(dir, f))))
        isempty(imgs) || size(img) == size(imgs[1]) ||
            throw(DimensionMismatch("$f has size $(size(img)), expected $(size(imgs[1]))"))
        push!(imgs, img)
        push!(names, name)
    end
    isempty(imgs) && throw(ArgumentError("no images in $dir"))
    X = Array{Float32}(undef, size(imgs[1])..., 1, length(imgs))
    for (k, img) in enumerate(imgs)
        X[:, :, 1, k] .= img
    end
    return X, names
end

"""
    flip_augment(rng, X)

Copy of the batch `X` (`H × W × C × N`) with every image mirrored horizontally with probability
1/2. No rotations or vertical flips: natural scenes have an up and a down.
"""
function flip_augment(rng::AbstractRNG, X::AbstractArray{T, 4}) where {T}
    Y = copy(X)
    for b in axes(X, 4)
        rand(rng, Bool) && (Y[:, :, :, b] .= X[:, end:-1:1, :, b])
    end
    return Y
end

"""
    to_model_range(x)
    from_model_range(y)

Images in `[0, 1]` ↔ the model range `[-1, 1]`.
"""
to_model_range(x) = 2 .* x .- 1
from_model_range(y) = (y .+ 1) ./ 2
