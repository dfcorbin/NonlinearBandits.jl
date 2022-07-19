function truncate_batch(limits::Matrix{Float64}, X::AbstractMatrix)
    X1 = deepcopy(X)
    n, d = size(X1)
    for i = 1:n, j = 1:d
        X1[i, j] = max(limits[j, 1], X1[i, j])
        X1[i, j] = min(limits[j, 2], X1[i, j])
    end
    return X1
end


mutable struct PolynomialThompsonSampling <: AbstractPolicy
    t::Int64
    batches::Int64
    X::Matrix{Float64}
    actions::Vector{Int64}
    rewards::Vector{Float64}
    arms::Vector{PartitionedPolyModel}
    initial_batches::Int64
    inflation::Float64
    retrain::Vector{Int64}

    limits::Matrix{Float64}
    limits_cache::Matrix{Float64}
    max_degree::Int64
    max_param::Int64
    max_models::Int64
    min_data::Int64
    regularization::Float64
    prior_shape::Float64
    prior_scale::Float64
    data_constraint::Float64
    tolerance::Float64
    verbose_retrain::Bool

    function PolynomialThompsonSampling(
        d::Int64,
        num_arms::Int64,
        initial_batches::Int64,
        retrain::Vector{Int64};
        inflation::Float64 = 1.0,
        max_degree::Int64 = 5,
        max_param::Int64 = 15,
        max_models::Int64 = 200,
        min_data::Int64 = 2,
        regularization::Float64 = 1.0,
        prior_shape::Float64 = 0.01,
        prior_scale::Float64 = 0.01,
        data_constraint::Float64 = 1.0,
        tolerance::Float64 = 1e-3,
        verbose_retrain::Bool = false,
    )
        limits = repeat([0.0 0.0], d, 1)
        limits_cache = deepcopy(limits)
        X = Matrix{Float64}(undef, 0, d)
        actions = Int64[]
        rewards = Float64[]
        arms = Vector{PartitionedPolyModel}(undef, num_arms)
        return new(
            0,
            0,
            X,
            actions,
            rewards,
            arms,
            initial_batches,
            inflation,
            retrain,
            limits,
            limits_cache,
            max_degree,
            max_param,
            max_models,
            min_data,
            regularization,
            prior_shape,
            prior_scale,
            data_constraint,
            tolerance,
            verbose_retrain,
        )
    end
end


function (pol::PolynomialThompsonSampling)(X::Matrix{Float64})
    n = size(X, 1)
    num_arms = length(pol.arms)
    actions = Vector{Int64}(undef, n)

    # Check if inital batches have been completed
    if pol.batches <= pol.initial_batches
        for i = 1:n
            actions[i] = (pol.t + i) % num_arms + 1
        end
        return actions
    end

    X1 = truncate_batch(pol.limits, X)
    thompson_samples = Vector{Float64}(undef, num_arms)
    for i = 1:n
        for a = 1:num_arms
            thompson_samples[a] = posterior_sample(pol.arms[a], X1[i, :], pol.inflation)
        end
        actions[i] = argmax(thompson_samples)
    end
    return actions
end


function update!(
    pol::PolynomialThompsonSampling,
    X::Matrix{Float64},
    actions::Vector{Int64},
    rewards::Vector{Float64},
)
    pol.t += size(X, 1)
    pol.batches += 1
    pol.X = vcat(pol.X, X)
    pol.actions = vcat(pol.actions, actions)
    pol.rewards = vcat(pol.rewards, rewards)

    for d = 1:size(X, 2)
        lower = minimum(X[:, d])
        upper = maximum(X[:, d])
        pol.limits_cache[d, 1] = min(lower, pol.limits_cache[d, 1])
        pol.limits_cache[d, 2] = max(upper, pol.limits_cache[d, 2])
    end

    if pol.batches < pol.initial_batches
        return nothing
    elseif pol.batches in pol.retrain || pol.batches == pol.initial_batches
        for a = 1:length(pol.arms)
            pol.limits = deepcopy(pol.limits_cache)
            Xa = pol.X[pol.actions.==a, :]
            ra = pol.rewards[pol.actions.==a]
            pol.arms[a] = PartitionedPolyModel(
                Xa,
                ra,
                pol.limits;
                max_degree = pol.max_degree,
                max_param = pol.max_param,
                max_models = pol.max_models,
                min_data = pol.min_data,
                data_constraint = pol.data_constraint,
                prior_shape = pol.prior_shape,
                prior_scale = pol.prior_scale,
                regularization = pol.regularization,
                tolerance = pol.tolerance,
                verbose = pol.verbose_retrain,
            )
        end
    else
        X1 = truncate_batch(pol.limits, X)
        for i in unique(actions)
            fit!(pol.arms[i], X1[actions.==i, :], rewards[actions.==i])
        end
    end
end


function predict(policy::PolynomialThompsonSampling, X::Matrix{Float64}, action::Int64)
    X1 = truncate_batch(policy.limits, X)
    return policy.arms[action](X1)
end