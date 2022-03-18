abstract type AbstractContextSampler end
abstract type AbstractDriver end
abstract type AbstractMetric end
abstract type AbstractPolicy end
abstract type AbstractRewardSampler end

mutable struct BanditDataset
    X::Matrix{Float64}
    a::Vector{Int64}
    r::Matrix{Float64}
end

function BanditDataset(d::Int64)
    if d <= 0
        throw(ArgumentError("d must be positive"))
    end
    X = Matrix{Float64}(undef, d, 0)
    a = Int64[]
    r = Matrix{Float64}(undef, 1, 0)
    return BanditDataset(X, a, r)
end

function arm_data(data::BanditDataset, a::Int64)
    if a <= 0
        throw(ArgumentError("a must be positive"))
    end
    idx = data.a .== a
    return data.X[:, idx], data.r[:, idx]
end

function add_data!(
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
