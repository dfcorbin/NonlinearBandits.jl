abstract type AbstractContextSampler end
abstract type AbstractRewardSampler end

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
struct GaussianRewards{T}
    mf::T
    σ::Float64

    function GaussianRewards(mf::T; σ::Float64=1.0) where {T<:Tuple{Vararg{<:Function}}}
        return new{T}(mf, σ)
    end
end

function (sampler::GaussianRewards)(X::AbstractMatrix, a::AbstractVector)
    n = size(X, 2)
    y = zeros(1, n)
    @inbounds for i in 1:n
        y[i] = sampler.mf[a[i]](X[:, i])
    end
    y += rand(Normal(0, sampler.σ), 1, n)
    return y
end