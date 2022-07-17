module NonlinearBandits

using Distributions: Uniform, Normal
using GLMNet: glmnet
using LinearAlgebra: cholesky, diagm, logdet, Symmetric
using Random: randperm
using SpecialFunctions: loggamma
using Suppressor: @suppress

include("LinearModel.jl")
include("PolyModel.jl")
include("PartitionedPolyModel.jl")
include("contexts.jl")
include("rewards.jl")
include("metrics.jl")
include("drivers.jl")

export AbstractContextSampler,
    AbstractDriver,
    AbstractMetric,
    AbstractPolicy,
    AbstractRewardSampler,
    append_data!,
    arm_data,
    BanditDataset,
    expand,
    fit!,
    FunctionalRegret,
    GaussianRewards,
    get_coefs,
    get_scale,
    get_shape,
    HyperParam,
    Index,
    LatentDriver,
    legendre_next,
    LinearModel,
    locate,
    Partition,
    PartitionedPolyModel,
    PolyModel,
    Region,
    run!,
    split!,
    StandardDriver,
    tensor_product_basis,
    UniformContexts,
    UniformPolicy,
    variance,
    WheelContexts,
    WheelRewards,
    WheelRegret

end
