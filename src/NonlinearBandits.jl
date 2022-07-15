module NonlinearBandits

using GLMNet: glmnet
using LinearAlgebra: cholesky, diagm, logdet, Symmetric
using Random: randperm
using SpecialFunctions: loggamma
using Suppressor: @suppress

include("LinearModel.jl")
include("PolyModel.jl")
include("PartitionedPolyModel.jl")

export expand,
    fit!,
    get_coefs,
    get_scale,
    get_shape,
    HyperParam,
    Index,
    legendre_next,
    tensor_product_basis,
    LinearModel,
    locate,
    Partition,
    PartitionedPolyModel,
    PolyModel,
    Region,
    split!,
    variance

end
