"""
Abstract type for models using a Gaussian/normal-inverse-gamma conjugate prior.
"""
abstract type AbstractBayesianLM end

"""
    BayesLM(d::Int; <keyword arguments>)        

Construct a Bayesian linear model.

# Arguments

- `λ::Float64=1.0`: Prior scaling.
- `shape0::Float64=1e-3`: Inverse-gamma prior shape hyperparameter.
- `scale0::Float64=1e-3`: Inverse-gamma prior scale hyperparameter.
"""
mutable struct BayesLM <: AbstractBayesianLM
    shape0::Float64
    scale0::Float64
    β0::Matrix{Float64}
    Σ0::Hermitian{Float64,Matrix{Float64}}
    Λ0::Hermitian{Float64,Matrix{Float64}}
    shape::Float64
    scale::Float64
    β::Matrix{Float64}
    Σ::Hermitian{Float64,Matrix{Float64}}
    Λ::Hermitian{Float64,Matrix{Float64}}

    function BayesLM(d::Int64; λ::Float64=1.0, shape0::Float64=1e-3, scale0::Float64=1e-3)
        if d <= 0 || shape0 <= 0 || scale0 <= 0
            msg = "d, shape0 and scale0 must be strictly positive"
            throw(ArgumentError(msg))
        end
        β0, Σ0, Λ0 = zeros(d, 1),
        Hermitian(diagm(ones(d)) * λ),
        Hermitian(diagm(ones(d)) / λ)
        β, Σ, Λ = deepcopy(β0), deepcopy(Σ0), deepcopy(Λ0)
        return new(shape0, scale0, β0, Σ0, Λ0, shape0, scale0, β, Σ, Λ)
    end
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

function check_regression_data(X::AbstractMatrix, y::AbstractMatrix)
    if size(y, 1) != 1
        throw(ArgumentError("y should have exactly 1 row"))
    elseif size(X, 2) != size(y, 2)
        throw(DimensionMismatch("X and y don't match in dimension 2"))
    end
end

function fit!(lm::BayesLM, X::AbstractMatrix, y::AbstractMatrix)
    check_regression_data(X, y)
    if size(lm.β, 1) != size(X, 1)
        msg = "X does not have the same number of predictors as the model"
        throw(DimensionMismatch(msg))
    end
    G = Hermitian(X * X')
    Λ = G + lm.Λ
    Σ = size(X, 2) == 1 ? Hermitian(sherman_morrison_inv(lm.Σ, X, X)) : inv(Λ)
    β = Σ * (X * y' + lm.Λ * lm.β)
    shape = lm.shape + size(y, 2) / 2
    scale = lm.scale + 0.5 * (y * y' - β' * Λ * β + lm.β' * lm.Λ * lm.β)[1, 1]
    if scale <= 0
        println("------------------")
        println(β' * Λ * β)
        println()
        println(lm.β' * lm.Λ * lm.β)
        println("------------------")
    end
    return lm.shape, lm.scale, lm.β, lm.Σ, lm.Λ = shape, scale, β, Σ, Λ
end

function (lm::BayesLM)(X::AbstractMatrix)
    if size(X, 1) != size(lm.β, 1)
        msg = "X does not have the same number of predictors as the model"
        throw(DimensionMismatch(msg))
    end
    return lm.β' * X
end

function shape_scale(lm::BayesLM)
    return lm.shape, lm.scale
end

"""
    std(model::AbstractBayesianLM)

Return the posterior mean of the inverse-gamma distribution.
"""
function std(model::AbstractBayesianLM)
    shape, scale = shape_scale(model)
    if shape <= 1
        throw(ArgumentError("normal-inverse-gamma has undefined mean for shape ≤ 1"))
    end
    return sqrt(scale / (shape - 1))
end
