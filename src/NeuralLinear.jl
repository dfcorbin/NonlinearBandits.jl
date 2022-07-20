function output_matrix(a::Vector{<:Int}, num_actions::Int)
    n = length(a)
    A = zeros(num_actions, n)
    for i = 1:n
        A[a[i], i] = 1.0
    end
    return A
end

mutable struct NeuralEncoder{T1<:Chain,T2<:Function,T3}
    num_inputs::Int64
    num_outputs::Int64
    layer_sizes::Vector{Int64}
    network::T1
    loss::T2
    pars::T3
end


function NeuralEncoder(
    num_inputs::Int,
    num_outputs::Int,
    layer_sizes::AbstractVector{<:Int},
)
    num_layers = length(layer_sizes)
    if num_layers < 1
        msg = "Must encode with at least 1 intermediate layer"
        throw(ArgumentError(msg))
    end
    encoding_layers = [Dense(num_inputs, layer_sizes[1], relu)]
    if num_layers > 1
        for i = 2:num_layers
            push!(encoding_layers, Dense(layer_sizes[i-1], layer_sizes[i], relu))
        end
    end
    enc = Chain(encoding_layers...) |> gpu
    dec = Dense(layer_sizes[end], num_outputs) |> gpu
    network = Chain(enc = enc, dec = dec)

    function loss(X::AbstractMatrix, A::AbstractMatrix, r::AbstractMatrix)
        # Subtract r[i] from each Y[i, :]
        # A[i, j] is 0/1 if action i was/wasn't taken at time j
        Y = (network(X) .- r) .* A
        return sum(Y .^ 2) / size(X, 2)
    end

    pars = params(network)
    return NeuralEncoder(num_inputs, num_outputs, layer_sizes, network, loss, pars)
end


function (encoder::NeuralEncoder)(X::AbstractMatrix)
    X1 = Matrix(X') |> gpu
    Z = encoder.network[:enc](X1) |> cpu
    return Matrix{Float64}(Z')
end


function predict(encoder::NeuralEncoder, X::AbstractMatrix)
    X1 = Matrix(X') |> gpu
    Y = encoder.network(X1) |> cpu
    return Matrix{Float64}(Y')
end


function fit!(
    encoder::NeuralEncoder,
    X::AbstractMatrix,
    actions::AbstractVector{<:Int64},
    rewards::AbstractVector,
    num_epochs::Int64;
    batch_size::Int64 = 32,
    opt::ADAM = ADAM(),
    verbose::Bool = true,
)

    X1 = Matrix(X') |> gpu
    A = output_matrix(actions, encoder.num_outputs) |> gpu
    r = reshape(rewards, (1, :)) |> Matrix |> gpu
    bs = min(size(X, 1), batch_size)
    data = DataLoader((X1, A, r), batchsize = bs, shuffle = true)
    for i = 1:num_epochs
        train!(encoder.loss, encoder.pars, data, opt)
        if verbose
            loss = encoder.loss(X1, A, r)
            print("\rEpoch $i\\$num_epochs: Training MSE = $loss")
        end
    end
end


mutable struct NeuralLinear{T1<:NeuralEncoder,T2} <: AbstractPolicy
    t::Int64
    batches::Int64
    Xs::Vector{Matrix{Float64}}
    Zs::Vector{Matrix{Float64}}
    rs::Vector{Vector{Float64}}
    X::Matrix{Float64}
    a::Vector{Int64}
    r::Vector{Float64}
    arms::Vector{LinearModel}
    layer_sizes::Vector{Int64}
    encoder::T1
    initial_batches::Int64
    retrain::Vector{Int64}
    num_epochs::Int64
    opt::T2
    inflation::Float64
    regularization::Float64
    prior_shape::Float64
    prior_scale::Float64
    verbose_retrain::Bool

    function NeuralLinear(
        d::Int64,
        num_arms::Int64,
        layer_sizes::Vector{Int64},
        inital_batches::Int64,
        retrain::Vector{Int64},
        num_epochs::Int64;
        opt = ADAM(),
        inflation::Float64 = 1.0,
        regularization::Float64 = 1.0,
        prior_shape::Float64 = 0.01,
        prior_scale::Float64 = 0.01,
        verbose_retrain::Bool = false,
    )
        arms = [
            LinearModel(
                layer_sizes[end],
                regularization = regularization,
                prior_shape = prior_shape,
                prior_scale = prior_scale,
            ) for a = 1:num_arms
        ]
        encoder = NeuralEncoder(d, num_arms, layer_sizes)
        Xs = [Matrix{Float64}(undef, 0, d) for a = 1:num_arms]
        Zs = [Matrix{Float64}(undef, 0, layer_sizes[end]) for a = 1:num_arms]
        rs = [Float64[] for a = 1:num_arms]
        X = Matrix{Float64}(undef, 0, d)
        a = Int64[]
        r = Float64[]

        return new{typeof(encoder),typeof(opt)}(
            0,
            0,
            Xs,
            Zs,
            rs,
            X,
            a,
            r,
            arms,
            layer_sizes,
            encoder,
            inital_batches,
            retrain,
            num_epochs,
            opt,
            inflation,
            regularization,
            prior_shape,
            prior_scale,
            verbose_retrain,
        )
    end
end


function update!(
    pol::NeuralLinear,
    X::AbstractMatrix,
    a::AbstractVector{<:Int},
    r::AbstractVector,
)
    pol.t += size(X, 1)
    pol.batches += 1
    for i in unique(a)
        Xa, ra = X[a.==i, :], r[a.==i]
        Za = pol.encoder(Xa)
        pol.Xs[i] = vcat(pol.Xs[i], Xa)
        pol.Zs[i] = vcat(pol.Zs[i], Za)
        pol.rs[i] = vcat(pol.rs[i], ra)
        pol.X = vcat(pol.X, X)
        pol.a = vcat(pol.a, a)
        pol.r = vcat(pol.r, r)
    end

    if pol.batches < pol.initial_batches
        return nothing
    end
    retrain_cond = pol.batches in pol.retrain || pol.batches == pol.initial_batches
    if retrain_cond
        pol.encoder = NeuralEncoder(size(X, 2), length(pol.arms), pol.layer_sizes)
        fit!(
            pol.encoder,
            pol.X,
            pol.a,
            pol.r,
            pol.num_epochs;
            opt = pol.opt,
            verbose = pol.verbose_retrain,
        )
        for i = 1:length(pol.arms)
            pol.Zs[i] = pol.encoder(pol.Xs[i])
        end
    end
    if pol.batches >= pol.initial_batches
        to_update = retrain_cond ? [1:length(pol.arms)...] : unique(a)
        for i in to_update
            pol.arms[i] = LinearModel(
                pol.layer_sizes[end],
                regularization = pol.regularization,
                prior_shape = pol.prior_shape,
                prior_scale = pol.prior_scale,
            )
            fit!(pol.arms[i], pol.Zs[i], pol.rs[i])
        end
    end
end


function (pol::NeuralLinear)(X::AbstractMatrix)
    n = size(X, 1)
    num_arms = length(pol.arms)
    actions = Vector{Int64}(undef, n)

    # Check if inital batches have been completed
    if pol.batches < pol.initial_batches
        for i = 1:n
            actions[i] = (pol.t + i) % num_arms + 1
        end
        return actions
    end

    Z = pol.encoder(X)
    thompson_samples = zeros(num_arms)
    for i = 1:n
        for a = 1:num_arms
            z = Z[i, :]
            thompson_samples[a] = posterior_sample(pol.arms[a], z, pol.inflation)
        end
        actions[i] = argmax(thompson_samples)
    end
    return actions
end