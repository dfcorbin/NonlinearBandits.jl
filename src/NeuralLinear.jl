function action_matrix(a::Vector{Int}, num_actions::Int)
    n = length(a)
    A = zeros(num_actions, n)
    for i in 1:n
        A[a[i], i] = 1.0
    end
    return A
end

"""
    NeuralEncoder(d_int::Int64, d_out::Int64, layer_sizes::Vector{Int64})

Construct a callable object that returns the final layer activations of a
neural network **(which can be trained with a contextual multi armed bandit trajectory)**.
"""
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
                push!(enc, Dense(layer_sizes[i-1], layer_sizes[i], relu))
            end
        end
        nn = Chain(enc=gpu(Chain(enc...)), dec=gpu(Dense(layer_sizes[end], d_out)))

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
    return convert(Matrix{Float64}, cpu(enc.nn[:enc](X)))
end

"""
    fit!(enc::NeuralEncoder, X::AbstractMatrix, a::AbstractVector{<:Int}, r::AbstractMatrix,
         epochs::Int64)
        
Update a the neural network parameters using a contextual multi-armed bandit trajectory.

# Keyword Arguments

- `batch_size::Int64=32`: The batch size to use while training the neural network.
- `opt=ADAM()`: The optimizer to use for updating the parameters.
- `verbose::Bool=true`: Print details of the fitting procedure.
"""
function fit!(
    enc::NeuralEncoder,
    X::AbstractMatrix,
    a::AbstractVector{<:Int},
    r::AbstractMatrix,
    epochs::Int64;
    batch_size::Int64=32,
    opt=ADAM(),
    verbose::Bool=true
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

"""
    NeuralLinear(d::Int64, num_arms::Int64, layer_sizes::Vector{Int64}, inital_batches::Int64,
                 retrain::Vector{Int64}, epochs::Int64)

NeuralLinear policy introduced in the paper [Deep Bayesian Bandits Showdown: An Empirical
Comparison of Bayesian Deep Networks for Thompson Sampling](https://arxiv.org/abs/1802.09127).

# Keyword Arguments

- `opt=ADAM()`: Optimizer used to update the neural network parameters.
- `λ::Float64=1.0`: Prior scaling.
- `shape0::Float64=1e-3`: Inverse-gamma prior shape hyperparameter.
- `scale0::Float64=1e-3`: Inverse-gamma prior scale hyperparameter.
- `verbose_retrain::Bool=false`: Print the details of the neural network training
    procedure.
"""
mutable struct NeuralLinear{T1<:NeuralEncoder,T2} <: AbstractPolicy
    t::Int64
    batches::Int64
    Xs::Vector{Matrix{Float64}}
    Zs::Vector{Matrix{Float64}}
    rs::Vector{Matrix{Float64}}
    X::Matrix{Float64}
    a::Vector{Int64}
    r::Matrix{Float64}
    arms::Vector{BayesLM}
    layer_sizes::Vector{Int64}
    enc::T1
    initial_batches::Int64
    retrain::Vector{Int64}
    epochs::Int64
    opt::T2
    α::Float64
    λ::Float64
    shape0::Float64
    scale0::Float64
    verbose_retrain::Bool

    function NeuralLinear(
        d::Int64,
        num_arms::Int64,
        layer_sizes::Vector{Int64},
        inital_batches::Int64,
        retrain::Vector{Int64},
        epochs::Int64;
        opt=ADAM(),
        α::Float64=1.0,
        λ::Float64=1.0,
        shape0::Float64=0.01,
        scale0::Float64=0.01,
        verbose_retrain::Bool=false
    )
        arms = [BayesLM(layer_sizes[end], λ=λ, shape0=shape0, scale0=scale0) for a in 1:num_arms]
        enc = NeuralEncoder(d, num_arms, layer_sizes)
        Xs = [Matrix{Float64}(undef, d, 0) for a in 1:num_arms]
        Zs = [Matrix{Float64}(undef, layer_sizes[end], 0) for a in 1:num_arms]
        rs = [Matrix{Float64}(undef, 1, 0) for a in 1:num_arms]
        X = Matrix{Float64}(undef, d, 0)
        a = Vector{Int64}(undef, 0)
        r = Matrix{Float64}(undef, 1, 0)

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
            α,
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
    check_regression_data(X, r)
    pol.t += size(X, 2)
    pol.batches += 1
    for i in unique(a)
        Xa, ra = X[:, a.==i], r[:, a.==i]
        Za = pol.enc(Xa)
        pol.Xs[i] = hcat(pol.Xs[i], Xa)
        pol.Zs[i] = hcat(pol.Zs[i], Za)
        pol.rs[i] = hcat(pol.rs[i], ra)
        pol.X = hcat(pol.X, X)
        pol.a = vcat(pol.a, a)
        pol.r = hcat(pol.r, r)
    end

    if pol.batches < pol.initial_batches
        return nothing
    end
    retrain_cond = pol.batches in pol.retrain || pol.batches == pol.initial_batches
    if retrain_cond
        pol.enc = NeuralEncoder(size(X, 1), length(pol.arms), pol.layer_sizes)
        fit!(
            pol.enc,
            pol.X,
            pol.a,
            pol.r,
            pol.epochs;
            opt=pol.opt,
            verbose=pol.verbose_retrain
        )
        for i in 1:length(pol.arms)
            pol.Zs[i] = pol.enc(pol.Xs[i])
        end
    end
    if pol.batches >= pol.initial_batches
        to_update = retrain_cond ? [1:length(pol.arms)...] : unique(a)
        for i in to_update
            pol.arms[i].Λ = Hermitian(pol.Zs[i] * pol.Zs[i]' + pol.arms[i].Λ0)
            pol.arms[i].Σ = inv(pol.arms[i].Λ)
            pol.arms[i].β = pol.arms[i].Σ * pol.Zs[i] * pol.rs[i]'
            pol.arms[i].shape = pol.arms[i].shape0 + size(pol.Zs[i], 2) / 2
            pol.arms[i].scale = pol.arms[i].scale0 + 0.5 * (
                pol.rs[i]*pol.rs[i]'-pol.arms[i].β'*pol.arms[i].Λ*pol.arms[i].β
            )[1, 1]
        end
    end
end

function (pol::NeuralLinear)(X::AbstractMatrix)
    n = size(X, 2)
    num_arms = length(pol.arms)
    actions = zeros(Int64, n)

    # Check if inital batches have been completed
    if pol.batches < pol.initial_batches
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
            Σ = pol.α * varsim * pol.arms[a].Σ
            local βsim
            try
                βsim = rand(MvNormal(β[:, 1], Σ))
            catch err
                @warn "Thompson sampling failed: $err"
                βsim = rand(Normal(0, 1), size(β))
            end
            thompson_samples[a] = βsim' * z
        end
        actions[i] = argmax(thompson_samples)
    end
    return actions
end

