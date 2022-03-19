"""
    GaussianRewards(mf::Tuple{Vararg{<:Function}}; <keyword arguments>)

Construct a callable object to sample gaussian rewards.

# Arguments
- `mf::Tuple{Vararg{<:Function}}`: A Tuple of functions which take a 1-dimensional input and
    output the (scalar) mean reward for the corresponding action.
- `σ::Float64`: The standard deviation of the gaussian noise applied to each reward. 
"""
struct GaussianRewards{T<:Tuple{Vararg{<:Function}}} <: AbstractRewardSampler
    mf::T
    σ::Float64

    function GaussianRewards(mf::T; σ::Float64=1.0) where {T<:Tuple{Vararg{<:Function}}}
        return new{T}(mf, σ)
    end
end

function (sampler::GaussianRewards)(X::AbstractMatrix, a::AbstractVector{<:Int})
    n = size(X, 2)
    r = zeros(1, n)
    for i in 1:n
        r[i] = sampler.mf[a[i]](X[:, i])
    end
    r += rand(Normal(0, sampler.σ), 1, n)
    return r
end
