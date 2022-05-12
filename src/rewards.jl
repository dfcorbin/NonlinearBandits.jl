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

struct WheelRewards <: AbstractRewardSampler
    δ::Float64
    normals::Tuple{Normal{Float64},Normal{Float64},Normal{Float64}}

    function WheelRewards(δ::Float64, μ::Tuple{Float64,Float64,Float64}, σ::Float64)
        if !(μ[3] >= μ[1] >= μ[2])
            throw(ArgumentError("Should have that μ[2] <= μ[1] <= μ[3]"))
        end
        normals = Tuple([Normal(μ[i], σ) for i in 1:3])
        return new(δ, normals)
    end
end

function (sampler::WheelRewards)(X::AbstractMatrix, a::AbstractVector{<:Int})
    n = length(a)
    r = zeros(1, n)
    for i in 1:n
        if a[i] == 1
            r[i] = rand(sampler.normals[1])
            continue
        end

        dst = sqrt(sum(X[:, i] .^ 2))
        if dst <= sampler.δ
            r[1, i] = rand(sampler.normals[2])
            continue
        end

        if (X[1, i] >= 0) && (X[2, i] >= 0)
            r[1, i] = a[i] == 2 ? rand(sampler.normals[3]) : rand(sampler.normals[2])
        elseif (X[1, i] >= 0) && (X[2, i] < 0)
            r[1, i] = a[i] == 3 ? rand(sampler.normals[3]) : rand(sampler.normals[2])
        elseif (X[1, i] < 0) && (X[2, i] <= 0)
            r[1, i] = a[i] == 4 ? rand(sampler.normals[3]) : rand(sampler.normals[2])
        elseif (X[1, i] < 0) && (X[2, i] < 0)
            r[1, i] = a[i] == 5 ? rand(sampler.normals[3]) : rand(sampler.normals[2])
        end
    end
    return r
end