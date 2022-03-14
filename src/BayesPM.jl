function legendre_next(x::Float64, j::Int64, p1::Float64, p0::Float64;
                       tol::Float64=1e-4)
    if x < -1.0 - tol || x > 1.0 + tol
        throw(DomainError("x lies outside [- 1 - 1e-4, 1 + 1e-4]"))
    elseif tol < 0.0
        throw(ArgumentError("tol must be non-negative"))
    end
    if j == 0 || j == 1
        return j == 0 ? 1.0 : x
    end
    x = min(x, 1.0)
    x = max(x, -1.0)
    return (2 * j - 1) * x * p1 / j - (j - 1) * p0 / j
end

"""
    Index(dim::Vector{Int64}, deg::Vector{Int64})

Multivariate monomial index.

The monomial `x[1] * x[3]^2`` can be encoded using `dim = [1, 3]`, `deg = [1, 2]`
"""
struct Index
    dim::Vector{Int64}
    deg::Vector{Int64}

    function Index(dim::Vector{Int64}, deg::Vector{Int64})
        if length(unique(dim)) != length(dim)
            throw(ArgumentError("dim contains repeated values"))
        elseif !all(dim .> 0)
            throw(ArgumentError("dim contains values < 1"))
        elseif !all(deg .>= 0)
            throw(ArgumentError("deg contains values < 0"))
        elseif length(dim) != length(deg)
            throw(DimensionMismatch("dim and deg have mismatched length"))
        else
            return new(dim, deg)
        end
    end
end

udeg(index) = length(index.deg) == 0 ? 0 : maximum(index.deg)

function fill_tpbasis!(d::Int64, J::Int64, basis::Vector{Index}, dim::Vector{Int64},
                       deg::Vector{Int64})
    # Reached the bottom of the recursion. Append the basis.
    if d == 0 || J == 0
        push!(basis, Index(dim, deg))
        return nothing
    end

    # Append the current basis function with every remaining j.
    for j in 0:J
        dim1, deg1 = deepcopy(dim), deepcopy(deg)
        if j > 0
            push!(dim1, d)
            push!(deg1, j)
        end
        fill_tpbasis!(d - 1, J - j, basis, dim1, deg1)
    end
end

"""
    tpbasis(d::Int64, J::Int64)

Construct the `d`-dimensional truncated tensor-product basis.

All index terms have a degree ≤ `J`.

See also [`Index`](@ref)
"""
function tpbasis(d::Int64, J::Int64)
    if d < 1
        throw(ArgumentError("recieved d < 1"))
    elseif J < 0
        throw(ArgumentError("recieved negative J"))
    end
    basis = Vector{Index}(undef, 0)
    dim, deg = Int64[], Int64[]
    fill_tpbasis!(d, J, basis, dim, deg)
    return basis
end

"""
    expand(X::AbstractMatrix, basis::Vector{Index}, limits::AbstractMatrix;
           J::Union{Nothing,Int64}=nothing)

Expand the columns of X into a rescaled legendre polynomial basis.

# Arguments
- `X::AbstractMatrix`: Matrix with observations stored as columns.
- `basis::Vector{<:Index}`: Vector of monomial indices.
- `limits::AbstractMatrix`: Matrix with two columns defining the lower/upper limits of the space.
- `J::Union{Nothing, Int64}=nothing`: The maximum degree of the basis. Inferred if not specified.
"""
function expand(X::AbstractMatrix, basis::Vector{Index}, limits::AbstractMatrix;
                J::Union{Nothing,Int64}=nothing)
    if size(limits) != (size(X, 1), 2)
        throw(ArgumentError("invalid expansion limits, expected size $((size(X, 1), 2))"))
    elseif !all(limits[:, 1] .<= limits[:, 2])
        throw(ArgumentError("limits[:, 1] must be <= limits[:, 2]"))
    elseif !isnothing(J) && J < 0
        throw(ArgumentError("J should be >= 0"))
    end

    J = isnothing(J) ? maximum(udeg.(basis)) : J # Infer J if not supplied
    d, n = size(X)
    nbas = length(basis)
    Y = ones(nbas, n)
    U = ones(J + 1, d)
    for i in 1:n
        # Univariate expansions
        for l in 1:d
            p1 = p0 = 1.0
            x = (2 * X[l, i] - limits[l, 1] - limits[l, 2]) / (limits[l, 2] - limits[l, 1])
            for j in 0:J
                U[j + 1, l] = legendre_next(x, j, p1, p0)
                p0 = p1
                p1 = U[j + 1, l]
            end
        end

        # Mulitplicative combinations
        for b in 1:nbas
            for l in 1:length(basis[b].dim)
                ud = basis[b].dim[l]
                uj = basis[b].deg[l]
                Y[b, i] *= U[uj + 1, ud]
            end
        end
    end
    return Y
end

mutable struct BayesPM
    J::Int64
    basis::Vector{Index}
    limits::Matrix{Float64}
    lm::BayesLM
end

"""
    BayesPM(basis::Vector{Index}, limits::AbstractMatrix; λ::Float64=1.0,
            shape0::Float64=1e-3, scale0::Float64=1e-3) 

Construct a Bayesian linear model on polynomial features.

# Arguments
- `basis::Vector{Index}`: Vector of monomial indices.
- `limits::AbstractMatrix`: Matrix with two columns defining the lower/upper limits of the space.
- `λ::Float64=1.0`: Prior covariance scale factor.
- `shape0::Float64=1e-3`: Inverse-gamma prior shape hyperparameter.
- `scale0::Float64=1e=3`: Inverse-gamma prior scale hyperparameter.
"""
function BayesPM(basis::Vector{Index}, limits::AbstractMatrix; λ::Float64=1.0,
                 shape0::Float64=1e-3, scale0::Float64=1e-3)
    lm = BayesLM(length(basis); λ=λ, shape0=shape0, scale0=scale0)
    J = maximum(udeg.(basis))
    return BayesPM(J, basis, limits, lm)
end

function fit!(pm::BayesPM, X::AbstractMatrix, y::AbstractMatrix)
    Z = expand(X, pm.basis, pm.limits; J=pm.J)
    return fit!(pm.lm, Z, y)
end

function (pm::BayesPM)(X::AbstractMatrix)
    Z = expand(X, pm.basis, pm.limits; J=pm.J)
    return pm.lm(Z)
end

std(pm::BayesPM) = std(pm.lm)