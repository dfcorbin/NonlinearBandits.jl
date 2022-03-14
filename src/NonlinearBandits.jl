module NonlinearBandits

using LinearAlgebra: Hermitian, diagm
using Distributions: Uniform, Normal

export fit!

"""
    fit!(model, X::AbstractMatrix, y::AbstractMatrix)

Update the parameters of `model`.

# Arguments

- `X::AbstractMatrix`: A matrix with observations stored as columns.
- `y::AbstractMatrix`: A matrix with 1 row of response variables. 
"""
function fit!(model, X::AbstractMatrix, y::AbstractMatrix) end

include("utils.jl")

export BayesLM, std
include("BayesLM.jl")

export Index, tpbasis, expand
include("BayesPM.jl")

end
