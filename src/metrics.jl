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
    met.regret = vcat(met.regret, regret)
    return nothing
end

mutable struct WheelRegret <: AbstractMetric
    δ::Float64
    μ::Tuple{Float64, Float64, Float64}
    regret::Vector{Float64}

    function WheelRegret(δ::Float64, μ::Tuple{Float64,Float64,Float64})
        if !(μ[3] >= μ[1] >= μ[2])
            throw(ArgumentError("Should have that μ[2] <= μ[1] <= μ[3]"))
        end
        return new(δ, μ, Float64[])
    end
end

function (met::WheelRegret)(X::AbstractMatrix, a::AbstractVector{<:Int}, r::AbstractMatrix)
    n = length(a)
    regret = zeros(n)
    for i in 1:n
        dst = sqrt(sum(X[:, i] .^ 2))
        if dst <= met.δ
            regret[i] = a[i] == 1 ? 0.0 : met.μ[1] - met.μ[2]
            continue
        end

        if (X[1, i] >= 0) && (X[2, i] >= 0)
            regret[i] = a[i] == 2 ? 0 : met.μ[3] - met.μ[2]
        elseif (X[1, i] >= 0) && (X[2, i] < 0)
            regret[i] = a[i] == 3 ? 0 : met.μ[3] - met.μ[2]
        elseif (X[1, i] < 0) && (X[2, i] <= 0)
            regret[i] = a[i] == 4 ? 0 : met.μ[3] - met.μ[2]
        elseif (X[1, i] < 0) && (X[2, i] < 0)
            regret[i] = a[i] == 5 ? 0 : met.μ[3] - met.μ[2]
        end
    end
    met.regret = vcat(met.regret, regret)
end