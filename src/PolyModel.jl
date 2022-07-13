function legendre_next(x::Float64, degree::Int64, p1::Float64, p0::Float64)
    if degree == 0 || degree == 1
        return degree == 0 ? 1.0 : x
    end
    return (2 * degree - 1) * x * p1 / degree - (degree - 1) * p0 / degree
end


struct Index
    dims::Vector{Int64}
    degrees::Vector{Int64}
end


function max_degree(idx::Index)
    return length(idx.dims) == 0 ? 0 : maximum(idx.degrees)
end


function _fill_tpbasis!(
    d::Int64,
    degree::Int64,
    basis::Vector{Index},
    dims::Vector{Int64},
    degrees::Vector{Int64},
)
    # Reached the bottom of the recursion. Append the basis.
    if d == 0 || degree == 0
        push!(basis, Index(dims, degrees))
        return nothing
    end

    # Append the current basis function with every remaining j.
    for j in 0:degree
        dims_cp, degrees_cp = deepcopy(dims), deepcopy(degrees)
        if j > 0
            push!(dims_cp, d)
            push!(degrees_cp, j)
        end
        _fill_tpbasis!(d - 1, degree - j, basis, dims_cp, degrees_cp)
    end
end


function tensor_product_basis(d::Int64, degree::Int64)
    basis = Index[]
    dims, degrees = Int64[], Int64[]
    _fill_tpbasis!(d, degree, basis, dims, degrees)
    return basis
end


function expand(
    X::AbstractMatrix,
    limits::Matrix{Float64},
    basis::Vector{Index},
    degree=nothing
)
    if size(limits) != (size(X, 2), 2)
        throw(ArgumentError("limits has incorrect size"))
    end

    degree = isnothing(degree) ? maximum(max_degree.(basis)) : degree
    n, d = size(X)
    num_bfuns = length(basis)
    Z = ones(n, num_bfuns)
    U = ones(d, degree + 1)
    for i in 1:n
        # Univariate expansions
        for l in 1:d
            p1 = p0 = 1.0
            x = (2 * X[i, l] - limits[l, 1] - limits[l, 2]) / (limits[l, 2] - limits[l, 1])
            for j in 0:degree
                U[l, j+1] = legendre_next(x, j, p1, p0)
                p0 = p1
                p1 = U[l, j+1]
            end
        end

        # Mulitplicative combinations
        for b in 1:num_bfuns
            for (dim, deg) in zip(basis[b].dims, basis[b].degrees)
                Z[i, b] *= U[dim, deg+1]
            end
        end
    end
    return Z
end


mutable struct PolyModel
    limits::Matrix{Float64}
    degree::Int64
    basis::Vector{Index}
    lm::LinearModel
end


function PolyModel(
    limits::Matrix{Float64},
    basis::Vector{Index};
    prior_shape::Float64=0.01,
    prior_scale::Float64=0.01,
    regularization::Float64=1.0
)
    lm = LinearModel(
        length(basis),
        prior_shape=prior_shape,
        prior_scale=prior_scale,
        regularization=regularization
    )
    degree = maximum(max_degree.(basis))
    return PolyModel(limits, degree, basis, lm)
end


function get_coefs(pm::PolyModel)
    return get_coefs(pm.lm)
end


function get_shape(pm::PolyModel)
    return get_shape(pm.lm)
end


function get_scale(pm::PolyModel)
    return get_scale(pm.lm)
end


function fit!(pm::PolyModel, X::Matrix{Float64}, y::Vector{Float64})
    Z = expand(X, pm.limits, pm.basis, pm.degree)
    fit!(pm.lm, Z, y)
    return nothing
end


function (pm::PolyModel)(X::Matrix{Float64})
    Z = expand(X, pm.limits, pm.basis, pm.degree)
    coefs = get_coefs(pm.lm)
    return Z * coefs
end


function variance(pm::PolyModel)
    shape = get_shape(pm.lm)
    scale = get_scale(pm.lm)
    return scale / (shape - 1)
end