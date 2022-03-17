module NonlinearBandits

using LinearAlgebra: Hermitian, diagm, logdet
using Distributions: Uniform, Normal, loggamma, InverseGamma, MvNormal
using Suppressor: @suppress
using GLMNet: glmnet
using Random: randperm

export fit!, update!

"""
    fit!(model, X::AbstractMatrix, y::AbstractMatrix)

Update the parameters of `model`.

# Arguments

- `X::AbstractMatrix`: A matrix with observations stored as columns.
- `y::AbstractMatrix`: A matrix with 1 row of response variables. 
"""
function fit!(model, X::AbstractMatrix, y::AbstractMatrix) end

include("utils.jl")

export BayesLM, std, AbstractBayesianLM
include("BayesLM.jl")

export Index, tpbasis, expand, BayesPM
include("BayesPM.jl")

export Partition,
    split!, locate, PartitionedBayesPM, auto_partitioned_bayes_pm, lasso_selection
include("PartitionedBayesPM.jl")

export AbstractContextSampler,
    AbstractRewardSampler,
    UniformContexts,
    GaussianRewards,
    AbstractPolicy,
    RandomPolicy,
    AbstractDriver,
    StandardDriver,
    BanditDataset,
    arm_data,
    add_data!,
    run!,
    FunctionalRegret
include("bandits.jl")

function update!(
    pol::AbstractPolicy, X::AbstractMatrix, a::AbstractVector{Int64}, r::AbstractMatrix
) end

export PolynomialThompsonSampling
include("PolynomialThompsonSampling.jl")

end