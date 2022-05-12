mutable struct GreedyPolynomial <: AbstractPolicy
    t::Int64
    batches::Int64
    data::BanditDataset
    arms::Vector{PartitionedBayesPM}
    initial_batches::Int64
    retrain::Vector{Int64}
    limits::Matrix{Float64}
    limits_cache::Matrix{Float64}

    Jmax::Int64
    Pmax::Int64
    Kmax::Int64
    λ::Float64
    shape0::Float64
    scale0::Float64
    ratio::Float64
    tol::Float64
    verbose_retrain::Bool

    function GreedyPolynomial(
        d::Int64,
        num_arms::Int64,
        initial_batches::Int64,
        retrain::Vector{Int64};
        Jmax::Int64=3,
        Pmax::Int64=100,
        Kmax::Int64=500,
        λ::Float64=1.0,
        shape0::Float64=1e-3,
        scale0::Float64=1e-3,
        ratio::Float64=1.0,
        tol::Float64=1e-3,
        verbose_retrain::Bool=false,
    )
        limits = repeat([0.0 0.0], d, 1)
        limits_cache = deepcopy(limits)
        data = BanditDataset(d)
        arms = Vector{PartitionedBayesPM}(undef, num_arms)
        return new(
            0,
            0,
            data,
            arms,
            initial_batches,
            retrain,
            limits,
            limits_cache,
            Jmax,
            Pmax,
            Kmax,
            λ,
            shape0,
            scale0,
            ratio,
            tol,
            verbose_retrain,
        )
    end
end

function (pol::GreedyPolynomial)(X::AbstractMatrix)
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

    X1 = truncate_batch(pol.limits, X)
    preds = zeros(num_arms)
    for i in 1:n
        for a in 1:num_arms
            preds[a] = X1[:, i:i][1, 1]
        end
        actions[i] = argmax(preds)
    end
    return actions
end

function update!(
    pol::GreedyPolynomial,
    X::AbstractMatrix,
    a::AbstractVector{<:Int},
    r::AbstractMatrix,
)
    pol.t += size(X, 2)
    pol.batches += 1
    append_data!(pol.data, X, a, r)

    for d in 1:size(X, 1)
        lower = minimum(X[d, :])
        upper = maximum(X[d, :])
        pol.limits_cache[d, 1] = min(lower, pol.limits_cache[d, 1])
        pol.limits_cache[d, 2] = max(upper, pol.limits_cache[d, 2])
    end

    if pol.batches < pol.initial_batches
        return nothing
    end
    if pol.batches in pol.retrain || pol.batches == pol.initial_batches
        for a in 1:length(pol.arms)
            pol.limits = deepcopy(pol.limits_cache)
            Xa, ra = arm_data(pol.data, a)
            pol.arms[a] = PartitionedBayesPM(
                Xa,
                ra,
                pol.limits;
                Jmax=pol.Jmax,
                Pmax=pol.Pmax,
                Kmax=pol.Kmax,
                λ=pol.λ,
                shape0=pol.shape0,
                scale0=pol.scale0,
                ratio=pol.ratio,
                tol=pol.tol,
                verbose=pol.verbose_retrain,
            )
        end
    else
        X1 = truncate_batch(pol.limits, X)
        for i in unique(a)
            fit!(pol.arms[i], X1[:, a .== i], r[:, a .== i])
        end
    end
end

function predict(policy::GreedyPolynomial, X::AbstractMatrix, a::Int64)
    X1 = truncate_batch(policy.limits, X)
    return policy.arms[a](X1)
end