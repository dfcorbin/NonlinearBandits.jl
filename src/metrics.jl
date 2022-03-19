"""
    FunctionalRegret(mf::Tuple{Vararg{<:Function}})

Metric to track the regret of each action using a discrete set of functions.
"""
mutable struct FunctionalRegret{T<:Tuple{Vararg{<:Function}}} <: AbstractMetric
    mf::T
    regret::Vector{Float64}
    function FunctionalRegret(mf::T) where {T<:Tuple{Vararg{<:Function}}}
        return new{T}(mf, Float64[])
    end
end

function (met::FunctionalRegret)(
    X::AbstractMatrix, a::AbstractVector{<:Int}, r::AbstractMatrix
)
    n = size(X, 2)
    Rmean = zeros(n, length(met.mf))
    for a in 1:length(met.mf)
        Rmean[:, a] = mapslices(met.mf[a], X; dims=1)
    end
    Rg = mapslices(row -> maximum(row) .- row, Rmean; dims=2)
    regret = [Rg[i, a[i]] for i in 1:n]
    return met.regret = vcat(met.regret, regret)
end