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
    return cpu(enc.nn[:enc](X))
end

function fit!(
    enc::NeuralEncoder,
    X::AbstractMatrix,
    a::AbstractVector{<:Int},
    r::AbstractMatrix,
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

mutable struct NeuralLinear{T1<:NeuralEncoder,T2} <: AbstractPolicy
    t::Int64
    batches::Int64
    data::BanditDataset
    arms::Vector{BayesLM}
    layer_sizes::Vector{Int64}
    enc::T1
    initial_batches::Int64
    retrain_freq::Int64
    epochs::Int64
    opt::T2
    λ::Float64
    shape0::Float64
    scale0::Float64
    verbose_retrain::Bool

    function NeuralLinear(
        d::Int64,
        num_arms::Int64,
        layer_sizes::Vector{Int64},
        inital_batches::Int64,
        retrain_freq::Int64,
        epochs::Int64;
        opt=ADAM(),
        λ::Float64=1.0,
        shape0::Float64=1e-3,
        scale0::Float64=1e-3,
        verbose_retrain::Bool=false,
    )
        arms = Vector{BayesLM}(undef, num_arms)
        enc = NeuralEncoder(d, num_arms, layer_sizes)
        data = BanditDataset(d)
        return new{typeof(enc),typeof(opt)}(
            0,
            0,
            data,
            arms,
            layer_sizes,
            enc,
            inital_batches,
            retrain_freq,
            epochs,
            opt,
            λ,
            shape0,
            scale0,
            verbose_retrain,
        )
    end
end

function update!(
    pol::NeuralLinear, X::AbstractMatrix, a::AbstractVector{<:Int}, r::AbstractMatrix
)
    pol.t += size(X, 2)
    pol.batches += 1
    append_data!(pol.data, X, a, r)
    if pol.batches < pol.initial_batches
        return nothing
    end
    if pol.batches % pol.retrain_freq == 0 || pol.batches == pol.initial_batches
        pol.enc = NeuralEncoder(size(X, 1), length(pol.arms), pol.layer_sizes)
        fit!(
            pol.enc,
            pol.data.X,
            pol.data.a,
            pol.data.r,
            pol.epochs;
            opt=pol.opt,
            verbose=pol.verbose_retrain,
        )
        for a in 1:length(pol.arms)
            Xa, ra = arm_data(pol.data, a)
            Za = pol.enc(Xa)
            pol.arms[a] = BayesLM(
                size(Za, 1); λ=pol.λ, shape0=pol.shape0, scale0=pol.scale0
            )
            fit!(pol.arms[a], Za, ra)
        end
    else
        Z = pol.enc(X)
        for i in unique(a)
            fit!(pol.arms[i], Z[:, a .== i], r[:, a .== i])
        end
    end
end

function (pol::NeuralLinear)(X::AbstractMatrix)
    n = size(X, 2)
    num_arms = length(pol.arms)
    actions = zeros(Int64, n)

    # Check if inital batches have been completed
    if pol.batches <= pol.initial_batches
        for i in 1:n
            actions[i] = (pol.t + i) % num_arms + 1
        end
        return actions
    end

    Z = pol.enc(X)
    thompson_samples = zeros(num_arms)
    for i in 1:n
        for a in 1:num_arms
            shape, scale = shape_scale(pol.arms[a])
            z = Z[:, i]
            varsim = rand(InverseGamma(shape, scale))
            β = pol.arms[a].β
            Σ = varsim * pol.arms[a].Σ
            βsim = rand(MvNormal(β[:, 1], Σ))
            thompson_samples[a] = βsim' * z
        end
        actions[i] = argmax(thompson_samples)
    end
    return actions
end
