mutable struct NeuralEncoder{T<:Chain}
    nn::T

    function NeuralEncoder(d_in::Int64, d_out::Int64, layer_sizes::Vector{Int64})
        if length(layer_sizes) < 1
            throw(ArgumentError("must provide at least 1 layer"))
        end
        enc = [Dense(d_in, layer_sizes[1], relu)]
        if length(layer_sizes) > 1
            for i in 2:length(layer_sizes)
                push!(enc, Dense(layer_sizes[i - 1], layer_sizes[i], relu))
            end
        end
        nn = Chain(; enc=Chain(enc...), dec=Dense(layer_sizes[end], d_out))
        return new{typeof(nn)}(nn)
    end
end

function (enc::NeuralEncoder)(X::AbstractArray)
    return enc.nn[:enc](X)
end