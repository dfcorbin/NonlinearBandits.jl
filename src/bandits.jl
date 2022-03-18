"""
Called with an integer `n > 0` to genrate a `d x n` matrix of contexts.
"""
abstract type AbstractContextSampler end

"""
Driver to manage how the bandit policy interacts with its environment.
"""
abstract type AbstractDriver end

"""
A metric that will be called as `metric(X, a, r)` after each batch outputted by a driver.
"""
abstract type AbstractMetric end

"""
A callable policy that outputs actions via `policy(X)` and is updated via [`update!`](@ref)
"""
abstract type AbstractPolicy end

"""
A callable object to ouput rewards via `sampler(X, a)`.
"""
abstract type AbstractRewardSampler end

"""
    BanditDataset(d::Int64)

Stores the trajectory of a bandit simulation.
"""
mutable struct BanditDataset
    X::Matrix{Float64}
    a::Vector{Int64}
    r::Matrix{Float64}

    function BanditDataset(d::Int64)
        if d <= 0
            throw(ArgumentError("d must be positive"))
        end
        X = Matrix{Float64}(undef, d, 0)
        a = Int64[]
        r = Matrix{Float64}(undef, 1, 0)
        return new(X, a, r)
    end
end

"""
    rm_data(data::BanditDataset, a::Int64)

Return the data associated with arm `a`.
"""
function arm_data(data::BanditDataset, a::Int64)
    if a <= 0
        throw(ArgumentError("a must be positive"))
    end
    idx = data.a .== a
    return data.X[:, idx], data.r[:, idx]
end

"""
    append_data!(data::BanditDataset, X::AbstractMatrix{Float64}, a::AbstractVector{Int64},
              r::AbstractMatrix{Float64})
            
Add a batch of data to the dataset.
"""
function append_data!(
    data::BanditDataset,
    X::AbstractMatrix{Float64},
    a::AbstractVector{Int64},
    r::AbstractMatrix{Float64},
)
    check_regression_data(X, r)
    if size(X, 1) != size(data.X, 1)
        throw(DimensionMismatch("X does not match the dimension of the dataset"))
    end
    data.X = hcat(data.X, X)
    data.r = hcat(data.r, r)
    return data.a = vcat(data.a, a)
end