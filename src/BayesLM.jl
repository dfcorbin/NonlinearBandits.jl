mutable struct BayesLM{T}
    shape0::T
    scale0::T
    β0::Matrix{T}
    Σ0::Hermitian{T,Matrix{T}}
    Λ0::Hermitian{T,Matrix{T}}
    shape::T
    scale::T
    β::Matrix{T}
    Σ::Hermitian{T,Matrix{T}}
    Λ::Hermitian{T,Matrix{T}}
end

function sherman_morrison_inv(Ainv::AbstractMatrix, u::AbstractMatrix, v::AbstractMatrix)
    if size(u) != size(v)
        throw(DimensionMismatch("u and v have mismatched dimensions"))
    elseif size(u, 2) != 1
        throw(ArgumentError("u and v should have exactly one column"))
    elseif size(Ainv) != (size(u, 1), size(u, 1))
        throw(DimensionMismatch("size of A does not match u/v"))
    end
    numer = Ainv * u * v' * Ainv
    denom = 1.0 + (v' * Ainv * u)[1, 1]
    return Ainv - numer / denom
end

"""
    BayesLM(d::Int; λ::AbstractFloat=1.0, shape0::AbsractFloat=1e-3,
            scale0::AbstractFloat=1e-3)
        
Construct a Bayesian linear model with prior scaling `λ`.
"""
function BayesLM(d::Int; λ::AbstractFloat=1.0, shape0::AbstractFloat=1e-3,
                 scale0::AbstractFloat=1e-3)
    if d <= 0 || shape0 <= 0 || scale0 <= 0
        throw(ArgumentError("d, shape0 and scale0 must be strictly positive"))
    end
    β0, Σ0, Λ0 = zeros(d, 1), Hermitian(diagm(ones(d)) * λ), Hermitian(diagm(ones(d)) / λ)
    β, Σ, Λ = deepcopy(β0), deepcopy(Σ0), deepcopy(Λ0)
    return BayesLM(shape0, scale0, β0, Σ0, Λ0, shape0, scale0, β, Σ, Λ)
end

function fit!(lm::BayesLM, X::AbstractMatrix, y::AbstractMatrix)
    if size(y, 1) != 1
        throw(ArgumentError("y should have exactly 1 row"))
    elseif size(X, 2) != size(y, 2)
        throw(DimensionMismatch("X and y don't match in dimension 2"))
    elseif size(lm.β, 1) != size(X, 1)
        throw(DimensionMismatch("X does not have the same number of predictors as the model"))
    end
    G = Hermitian(X * X')
    Λ = G + lm.Λ
    Σ = size(X, 2) == 1 ? Hermitian(sherman_morrison_inv(lm.Σ, X, X)) : inv(Λ)
    β = Σ * (X * y' + lm.Λ * lm.β)
    shape = lm.shape + size(y, 2) / 2
    scale = lm.scale + 0.5 * (y * y' - β' * Λ * β + lm.β' * lm.Λ * lm.β)[1, 1]
    return lm.shape, lm.scale, lm.β, lm.Σ, lm.Λ = shape, scale, β, Σ, Λ
end

"""
    std(lm::BayesLM)

Computed the posterior expected standard deviation from the model lm.
"""
function std(lm::BayesLM)
    if lm.shape <= 1
        throw(ArgumentError("expected variance is undefined for shape ≤ 1"))
    end
    return sqrt(lm.scale / (lm.shape - 1))
end

function (lm::BayesLM)(X::AbstractMatrix)
    if size(X, 1) != size(lm.β, 1)
        throw(ArgumentError("X does not have the same number of predictors as the model"))
    end
    return lm.β' * X
end
