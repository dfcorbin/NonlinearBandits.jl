abstract type AbstractRewardSampler end


struct GaussianRewards{T<:Tuple{Vararg{<:Function}}} <: AbstractRewardSampler
    mean_funs::T
    noise::Normal{Float64}

    function GaussianRewards(mean_funs::Tuple{Vararg{<:Function}}, sd::Float64)
        noise = Normal(0, sd)
        return new{typeof(mean_funs)}(mean_funs, noise)
    end
end


function (sampler::GaussianRewards)(X::Matrix{Float64}, actions::Vector{Int64})
    rewards = Vector{Float64}(undef, size(X, 1))
    for (i, (a, x)) in enumerate(zip(actions, eachrow(X)))
        rewards[i] = sampler.mean_funs[a](x) + rand(sampler.noise)
    end
    return rewards
end


struct WheelRewards <: AbstractRewardSampler
    radius::Float64
    normals::Tuple{Normal{Float64},Normal{Float64},Normal{Float64}}

    function WheelRewards(
        radius::Float64,
        means::Tuple{Float64,Float64,Float64},
        sd::Float64,
    )
        if !(means[3] > means[1] > means[2])
            throw(ArgumentError("Incorrect mean ordering"))
        end
        normals = Tuple([Normal(m, sd) for m in means])
        return new(radius, normals)
    end
end


function (sampler::WheelRewards)(X::Matrix{Float64}, actions::Vector{Int64})
    n = size(X, 1)
    rewards = Vector{Float64}(undef, n)
    for i = 1:n
        if actions[i] == 1
            rewards[i] = rand(sampler.normals[1])
        elseif sqrt(sum(X[i, :] .^ 2)) <= sampler.radius
            rewards[i] = rand(sampler.normals[2])
        elseif (actions[i] == 2 && X[i, 1] >= 0 && X[i, 2] >= 0) ||
               (actions[i] == 3 && X[i, 1] >= 0 && X[i, 2] < 0) ||
               (actions[i] == 4 && X[i, 1] < 0 && X[i, 2] >= 0) ||
               (actions[i] == 5 && X[i, 1] < 0 && X[i, 2] < 0)
            rewards[i] = rand(sampler.normals[3])
        else
            rewards[i] = rand(sampler.normals[2])
        end
    end
    return rewards
end