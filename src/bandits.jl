abstract type AbstractContextSampler end
abstract type AbstractDriver end
abstract type AbstractMetric end
abstract type AbstractPolicy end
abstract type AbstractRewardSampler end

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

function (pol::RandomPolicy)(X::AbstractMatrix{Float64})
    n = size(X, 2)
    return rand(1:(pol.num_actions), n)
end

mutable struct BanditDataset
    X::Matrix{Float64}
    a::Vector{Int64}
    r::Matrix{Float64}
end

function BanditDataset(d::Int64)
    X = Matrix{Float64}(undef, d, 0)
    a = Int64[]
    r = Matrix{Float64}(undef, 1, 0)
    return BanditDataset(X, a, r)
end

function arm_data(data::BanditDataset, a::Int64)
    idx = data.a .== a
    return data.X[:, idx], data.r[:, idx]
end

function add_data!(
    data::BanditDataset,
    X::AbstractMatrix{Float64},
    a::AbstractVector{Int64},
    r::AbstractMatrix{Float64},
)
    data.X = hcat(data.X, X)
    data.r = hcat(data.r, r)
    return data.a = vcat(data.a, a)
end
