function action_matrix(a::Vector{Int}, num_actions::Int)
    n = length(a)
    A = zeros(num_actions, n)
    for i in 1:n
        A[a[i], i] = 1.0
    end
    return A
end

mutable struct NeuralEncoder{T1<:Chain,T2<:Function}
    nn::T1
    loss::T2
    d_in::Int64
    d_out::Int64
    layer_sizes::Vector{Int64}

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
        nn = Chain(; enc=gpu(Chain(enc...)), dec=gpu(Dense(layer_sizes[end], d_out)))

        # Create partial loss function
        function loss(X::AbstractMatrix, A::AbstractMatrix, r::AbstractMatrix)
            Ypred = nn(X)
            mse = sum(((Ypred .- r) .* A) .^ 2) / size(r, 2)
            return mse
        end

        return new{typeof(nn),typeof(loss)}(nn, loss, d_in, d_out, layer_sizes)
    end
end

function (enc::NeuralEncoder)(X::AbstractArray)
    X = gpu(X)
    return enc.nn[:enc](X)
end

function fit!(
    enc::NeuralEncoder,
    X::AbstractMatrix{Float64},
    a::AbstractVector{Int64},
    r::AbstractMatrix{Float64},
    epochs::Int64;
    batch_size::Int64=32,
    opt=ADAM(),
    verbose::Bool=true,
)
    check_regression_data(X, r)
    if length(a) != size(X, 2)
        throw(ArgumentError("length of a must equal the second dimension of X/r"))
    end
    X = gpu(X)
    A = gpu(action_matrix(a, enc.d_out))
    r = gpu(r)
    data = DataLoader((X, A, r); batchsize=batch_size, shuffle=true)
    for i in 1:epochs
        train!(enc.loss, params(enc.nn), data, opt)
        if verbose
            println("\rEpoch: ", i, "; Training loss: ", enc.loss(X, A, r))
        end
    end
end
