abstract type AbstractPolicy end


struct UniformPolicy <: AbstractPolicy
    num_actions::Int64
end


function (pol::UniformPolicy)(X::Matrix{Float64})
    return rand(1:pol.num_actions, size(X, 1))
end


function update!(
    pol::UniformPolicy,
    X::Matrix{Float64},
    actions::Vector{Int64},
    rewards::Vector{Float64},
)
    return nothing
end


abstract type AbstractDriver end


mutable struct StandardDriver{
    T1<:AbstractContextSampler,
    T2<:AbstractRewardSampler,
    T3<:AbstractPolicy,
} <: AbstractDriver
    csampler::T1
    policy::T3
    rsampler::T2
    metrics::Vector{AbstractMetric}
end


function StandardDriver(
    csampler::AbstractContextSampler,
    policy::AbstractPolicy,
    rsampler::AbstractRewardSampler,
)
    return StandardDriver(csampler, policy, rsampler, AbstractMetric[])
end


function (driver::StandardDriver)(batch_size::Int64)
    X = driver.csampler(batch_size)
    actions = driver.policy(X)
    rewards = driver.rsampler(X, actions)
    for met in driver.metrics
        met(X, actions, rewards)
    end
    return X, actions, rewards
end


mutable struct LatentDriver{
    T1<:AbstractContextSampler,
    T2<:AbstractRewardSampler,
    T3<:AbstractPolicy,
} <: AbstractDriver
    csampler::T1
    policy::T3
    rsampler::T2
    tform::Function
    metrics::Vector{AbstractMetric}
end


function LatentDriver(
    csampler::AbstractContextSampler,
    policy::AbstractPolicy,
    rsampler::AbstractRewardSampler,
    tform::Function,
)
    return LatentDriver(csampler, policy, rsampler, tform, ())
end


function (driver::LatentDriver)(batch_size::Int64)
    Z = driver.csampler(batch_size)
    X = mapslices(driver.tform, Z; dims = 2)
    actions = driver.policy(X)
    rewards = driver.rsampler(Z, actions)
    for met in driver.metrics
        met(Z, actions, rewards)
    end
    return X, actions, rewards
end


function run!(
    num_batches::Int64,
    batch_size::Int64,
    driver::AbstractDriver;
    verbose::Bool = true,
)
    for i = 1:num_batches
        if verbose
            print("\rBatch $i/$num_batches")
        end
        X, actions, rewards = driver(batch_size)
        update!(driver.policy, X, actions, rewards)
    end
end