module NonlinearBandits

using Flux:
    relu, Chain, Dense, gpu, @epochs, train!, ADAM, DataLoader, throttle, params, cpu
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
include("NeuralLinear.jl")

"""
    fit!(model, X::AbstractMatrix, y::AbstractMatrix)

Update the parameters of `model`.

# Arguments

- `X::AbstractMatrix`: A matrix with observations stored as columns.
- `y::AbstractMatrix`: A matrix with 1 row of response variables. 
"""
function fit!(model, X::AbstractMatrix, y::AbstractMatrix)
    throw(ErrorException("must implement fit! for $(typeof(model))"))
end

"""
    shape_scale(model::AbstractBayesianLM)

Return the shape/scale of model.
"""
function shape_scale(model::AbstractBayesianLM)
    throw(ErrorException("must implement shape_scale for $(typeof(model))"))
end

"""
    update!(pol::AbstractPolicy, X::AbstractMatrix, a::AbstractVector{<:Int}, 
            r::AbstractMatrix) 

Update `pol` with a batch of data.
"""
function update!(
    pol::AbstractPolicy, X::AbstractMatrix, a::AbstractVector{<:Int}, r::AbstractMatrix
)
    throw(ErrorException("must implement update! for $(typeof(pol))"))
end

export AbstractBayesianLM,
    AbstractContextSampler,
    AbstractDriver,
    AbstractMetric,
    AbstractPolicy,
    AbstractRewardSampler,
    append_data!,
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
    LatentDriver,
    locate,
    NeuralEncoder,
    NeuralLinear,
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

end
