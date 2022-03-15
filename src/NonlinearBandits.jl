module NonlinearBandits

using LinearAlgebra: Hermitian, diagm, logdet
using Distributions: Uniform, Normal, loggamma
using Random: randperm

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

export Index, tpbasis, expand, BayesPM
include("BayesPM.jl")

export Partition, split!, locate, PartitionedBayesPM, auto_partitioned_bayes_pm
include("PartitionedBayesPM.jl")

end
