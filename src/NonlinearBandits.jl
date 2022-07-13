module NonlinearBandits

using LinearAlgebra: cholesky, diagm, Symmetric

include("LinearModel.jl")

export
    fit!,
    get_coefs,
    get_scale,
    get_shape,
    HyperParam,
    LinearModel,
    variance

end
