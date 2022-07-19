function action_matrix(a::Vector{<:Int}, num_actions::Int)
    n = length(a)
    A = zeros(num_actions, n)
    for i = 1:n
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
            for i = 2:length(layer_sizes)
                push!(enc, Dense(layer_sizes[i-1], layer_sizes[i], relu))
            end
        end
        nn = Chain(enc = gpu(Chain(enc...)), dec = gpu(Dense(layer_sizes[end], d_out)))

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
    X = gpu(X')
    return convert(Matrix{Float64}, cpu(enc.nn[:enc](X))')
end


function fit!(
    enc::NeuralEncoder,
    X::AbstractMatrix,
    a::AbstractVector{<:Int},
    r::AbstractVector,
    epochs::Int64;
    batch_size::Int64 = 32,
    opt = ADAM(),
    verbose::Bool = true,
)

    X = gpu(X')
    A = gpu(action_matrix(a, enc.d_out))
    r = gpu(reshape(r, (1, :)))
    data = DataLoader((X, A, r); batchsize = batch_size, shuffle = true)
    for i = 1:epochs
        train!(enc.loss, params(enc.nn), data, opt)
        if verbose
            println("\rEpoch: ", i, "; Training loss: ", enc.loss(X, A, r))
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
    enc::T1
    initial_batches::Int64
    retrain::Vector{Int64}
    epochs::Int64
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
        epochs::Int64;
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
        enc = NeuralEncoder(d, num_arms, layer_sizes)
        Xs = [Matrix{Float64}(undef, d, 0) for a = 1:num_arms]
        Zs = [Matrix{Float64}(undef, layer_sizes[end], 0) for a = 1:num_arms]
        rs = [Float64[] for a = 1:num_arms]
        X = Matrix{Float64}(undef, d, 0)
        a = Vector{Int64}
        r = Float64[]

        return new{typeof(enc),typeof(opt)}(
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
            enc,
            inital_batches,
            retrain,
            epochs,
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
        Za = pol.enc(Xa)
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
        pol.enc = NeuralEncoder(size(X, 2), length(pol.arms), pol.layer_sizes)
        fit!(
            pol.enc,
            pol.X,
            pol.a,
            pol.r,
            pol.epochs;
            opt = pol.opt,
            verbose = pol.verbose_retrain,
        )
        for i = 1:length(pol.arms)
            pol.Zs[i] = pol.enc(pol.Xs[i])
        end
    end
    if pol.batches >= pol.initial_batches
        to_update = retrain_cond ? [1:length(pol.arms)...] : unique(a)
        for i in to_update
            pol.arms[i] = LinearModel(
                layer_sizes[end],
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

    Z = pol.enc(X)
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