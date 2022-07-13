module NonlinearBandits

using LinearAlgebra: cholesky, diagm, Symmetric

include("LinearModel.jl")
include("PolyModel.jl")
include("PartitionedPolyModel.jl")

export
    expand,
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
    PolyModel,
    Region,
    split!,
    variance

end
