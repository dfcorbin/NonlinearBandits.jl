abstract type AbstractContextSampler end
abstract type AbstractRewardSampler end
abstract type AbstractPolicy end
abstract type AbstractDriver end

"""
    UniformContexts(limits::Matrix{Float64})

Construct a callable object to generate uniform contexts.
"""
struct UniformContexts <: AbstractContextSampler
    limits::Matrix{Float64}

    function UniformContexts(limits::Matrix{Float64})
        check_limits(limits)
        return new(limits)
    end
end

function (sampler::UniformContexts)(n::Int64)
    d = size(sampler.limits, 1)
    X = zeros(d, n)
    for i in 1:d
        X[i, :] = rand(Uniform(sampler.limits[i, 1], sampler.limits[i, 2]), n)
    end
    return X
end

"""
    GaussianRewards(mf::T; <keyword arguments>) where {T<:Tuple{Vararg{<:Function}}}

Construct a callable object to sample gaussian rewards.

# Arguments
- `mf`: A Tuple of functions which take a 1-dimensional input and output the (scalar)
        mean reward for the corresponding action.
- `σ::Float64`: The standard deviation of the gaussian noise applied to each reward. 
"""
struct GaussianRewards{T} <: AbstractRewardSampler
    mf::T
    σ::Float64

    function GaussianRewards(mf::T; σ::Float64=1.0) where {T<:Tuple{Vararg{<:Function}}}
        return new{T}(mf, σ)
    end
end

function (sampler::GaussianRewards)(X::AbstractMatrix, a::AbstractVector)
    n = size(X, 2)
    r = zeros(1, n)
    @inbounds for i in 1:n
        r[i] = sampler.mf[a[i]](X[:, i])
    end
    r += rand(Normal(0, sampler.σ), 1, n)
    return r
end

"""
    RandomPolicy(num_actions::Int64)

Construct a policy that chooses actions at random.
"""
struct RandomPolicy <: AbstractPolicy
    num_actions::Int64
    function RandomPolicy(num_actions::Int64)
        return if num_actions > 0
            new(num_actions)
        else
            throw(ArgumentError("num_actions must be positive"))
        end
    end
end

function (pol::RandomPolicy)(X::AbstractMatrix)
    n = size(X, 2)
    return rand(1:(pol.num_actions), n)
end

"""
    StandardDriver(csampler::AbstractContextSampler, policy::AbstractPolicy, 
                   rsampler::AbstractRewardSampler[, metrics::Tuple])

A simple driver that samples contexts, passes them to the policy to generate actions,
then observes the rewards.

# Arguments
- `csampler::AbstractContextSampler`: Context sampler.
- `policy::AbstractPolicy`: Policy to generate actions given contexts.
- `rsampler::AbstractRewardSampler`: Sampler for rewards, given the contexts and actions.
- `metrics`: A tuple of callable objects that will each be called obj(X, a, r).
"""
mutable struct StandardDriver{
    T1<:AbstractContextSampler,T2<:AbstractRewardSampler,T3<:AbstractPolicy,T4<:Tuple
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

function (driver::StandardDriver)(n::Int64)
    X = driver.csampler(n)
    a = driver.policy(X)
    r = driver.rsampler(X, a)
    for m in driver.metrics
        m(X, a, r)
    end
    return X, a, r
end