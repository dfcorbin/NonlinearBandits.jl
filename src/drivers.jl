"""
    StandardDriver(csampler::AbstractContextSampler, policy::AbstractPolicy, 
                   rsampler::AbstractRewardSampler[, 
                   metrics::Tuple{Vararg{<:AbstractMetric}}])

A simple driver that samples contexts, passes them to the policy to generate actions,
then observes the rewards.

# Arguments
- `csampler::AbstractContextSampler`: Context sampler.
- `policy::AbstractPolicy`: Policy to generate actions given contexts.
- `rsampler::AbstractRewardSampler`: Sampler for rewards, given the contexts and actions.
- `metrics::Tuple{Vararg{<:AbstractMetric}}`: A tuple of metrics that will each be 
    called as `metric(X, a, r)`.
"""
mutable struct StandardDriver{
    T1<:AbstractContextSampler,
    T2<:AbstractRewardSampler,
    T3<:AbstractPolicy,
    T4<:Tuple{Vararg{<:AbstractMetric}},
} <: AbstractDriver
    csampler::T1
    policy::T3
    rsampler::T2
    metrics::T4
end

function StandardDriver(
    csampler::AbstractContextSampler,
    policy::AbstractPolicy,
    rsampler::AbstractRewardSampler,
)
    return StandardDriver(csampler, policy, rsampler, ())
end

function (driver::StandardDriver)(batchsize::Int64)
    X = driver.csampler(batchsize)
    a = driver.policy(X)
    r = driver.rsampler(X, a)
    for met in driver.metrics
        met(X, a, r)
    end
    return X, a, r
end

"""
    run!(num_batches::Int64, batch_size::Int64, driver::AbstractDriver; <keyword arguments>)

Run a `driver` for `num_batches` batches.
"""
function run!(
    num_batches::Int64, batch_size::Int64, driver::AbstractDriver; verbose::Bool=true
)
    if num_batches <= 0 || batch_size <= 0
        throw(ArgumentError("num_batches and batch_size must be positive"))
    end
    for i in 1:num_batches
        if verbose
            print("\rStep $i/$num_batches")
        end
        X, a, r = driver(batch_size)
        update!(driver.policy, X, a, r)
    end
end