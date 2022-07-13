module NonlinearBandits

using LinearAlgebra: cholesky, diagm, Symmetric

include("LinearModel.jl")
include("PolyModel.jl")

export
    fit!,
    get_coefs,
    get_scale,
    get_shape,
    HyperParam,
    legendre,
    LinearModel,
    Polynomial,
    variance

end
