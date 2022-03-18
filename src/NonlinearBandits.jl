module NonlinearBandits

using Distributions: Uniform, Normal, loggamma, InverseGamma, MvNormal
using GLMNet: glmnet
using LinearAlgebra: Hermitian, diagm, logdet
using Random: randperm
using Suppressor: @suppress

include("utils.jl")
include("BayesLM.jl")
include("BayesPM.jl")
include("PartitionedBayesPM.jl")
include("bandits.jl")
include("contexts.jl")
include("rewards.jl")
include("metrics.jl")
include("drivers.jl")
include("RandomPolicy.jl")
include("PolynomialThompsonSampling.jl")

"""
    fit!(model, X::AbstractMatrix{Float64}, y::AbstractMatrix{Float64})

Update the parameters of `model`.

# Arguments

- `X::AbstractMatrix{Float64}`: A matrix with observations stored as columns.
- `y::AbstractMatrix{Float64}`: A matrix with 1 row of response variables. 
"""
function fit!(model, X::AbstractMatrix, y::AbstractMatrix) end

"""
    shape_scale(model::AbstractBayesianLM)

Return the shape/scale of model.
"""
function shape_scale(model::AbstractBayesianLM) end

"""
    update!(pol::AbstractPolicy, X::AbstractMatrix{Float64}, a::AbstractVector{Int64}, 
            r::AbstractMatrix{Float64}) 

Update `pol` with a batch of data.
"""
function update!(
    pol::AbstractPolicy,
    X::AbstractMatrix{Float64},
    a::AbstractVector{Int64},
    r::AbstractMatrix{Float64},
) end

export AbstractBayesianLM,
    AbstractContextSampler,
    AbstractDriver,
    AbstractMetric,
    AbstractPolicy,
    AbstractRewardSampler,
    add_data!,
    arm_data,
    BanditDataset,
    BayesLM,
    BayesPM,
    expand,
    fit!,
    FunctionalRegret,
    GaussianRewards,
    Index,
    lasso_selection,
    locate,
    Partition,
    PartitionedBayesPM,
    PolynomialThompsonSampling,
    RandomPolicy,
    run!,
    shape_scale,
    split!,
    StandardDriver,
    std,
    tpbasis,
    UniformContexts,
    update!

export AbstractContextSampler,
    AbstractRewardSampler,
    UniformContexts,
    GaussianRewards,
    AbstractPolicy,
    RandomPolicy,
    AbstractDriver,
    StandardDriver
include("bandits.jl")

end
