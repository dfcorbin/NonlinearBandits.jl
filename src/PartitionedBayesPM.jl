"""
    Partition(limits::Matrix{Float64})

Construct an object capable of storing a hyperrectangular partition.
"""
mutable struct Partition
    limits::Matrix{Float64}
    regions::Vector{Matrix{Float64}}

    function Partition(limits::Matrix{Float64})
        check_limits(limits)
        return new(deepcopy(limits), deepcopy([limits]))
    end
end

"""
    split!(P::Partition, k::Int64, d::Int64) 

Split the `k`'th subregion of `P` into equal halves in dimension `d`.
"""
function split!(P::Partition, k::Int64, d::Int64)
    if !(1 <= k <= length(P.regions))
        throw(ArgumentError("P does not contain a subregion at index $k"))
    elseif !(1 <= d <= size(P.limits, 1))
        throw(ArgumentError("P is not $d-dimensional"))
    end
    right = deepcopy(P.regions[k])
    loc = sum(right[d, :]) / 2
    P.regions[k][d, 2] = loc
    right[d, 1] = loc
    return push!(P.regions, right)
end

"""
    locate(P::Partition, X::AbstractMatrix)

Return a vector of integers giving the region index for each column of `X`.
"""
function locate(P::Partition, X::AbstractMatrix)
    d, n = size(X)
    if d != size(P.limits, 1)
        throw(DimensionMismatch("P does match the dimension of X"))
    end
    idx = Vector{Int64}(undef, n)
    for i in 1:n
        contained = true
        for k in 1:length(P.regions)
            for l in 1:d
                upper = X[l, i] == P.regions[k][l, 2] == P.limits[l, 2]
                below = X[l, i] >= P.regions[k][l, 1]
                above = X[l, i] < P.regions[k][l, 2]
                contained = below && (above || upper)
                if !contained
                    break
                end
            end
            if contained
                idx[i] = k
                break
            end
        end
        if !contained
            throw(ArgumentError("no region located for observation $i"))
        end
    end
    return idx
end

mutable struct PartitionedBayesPM <: AbstractBayesianLM
    P::Partition
    models::Vector{BayesPM}
    shape0::Float64
    scale0::Float64
    shape::Float64
    scale::Float64

    function PartitionedBayesPM(
        P::Partition, models::Vector{BayesPM}, shape0::Float64, scale0::Float64
    )
        shape = shape0
        scale = scale0
        for pm in models
            shape += pm.lm.shape - shape0
            scale += pm.lm.scale - scale0
        end
        return new(P, models, shape0, scale0, shape, scale)
    end
end

function shape_scale(ppm::PartitionedBayesPM)
    return ppm.shape, ppm.scale
end

"""
    PartitionedBayesPM(P::Partition, Js::Vector{Int64}; <keyword arguments>)

Contruct a partitioned polynomial model.

# Arguments

- `P::Partition`: A partition of the space.
- `λ::Float64=1.0`: Prior scaling.
- `shape0::Float64=1e-3`: Inverse-gamma prior shape hyperparameter.
- `scale0::Float64=1e-3`: Inverse-gamma prior scale hyperparameter.
"""
function PartitionedBayesPM(
    P::Partition,
    Js::Vector{Int64};
    λ::Float64=1.0,
    shape0::Float64=0.01,
    scale0::Float64=0.01
)
    if length(P.regions) != length(Js)
        throw(ArgumentError("must supply a value of J for every region in P"))
    end
    d = size(P.limits, 1)
    basis = [tpbasis(d, J) for J in Js]
    models = [
        BayesPM(basis[i], P.regions[i]; λ=λ, shape0=shape0, scale0=scale0) for
        i in 1:length(Js)
    ]
    return PartitionedBayesPM(P, models, shape0, scale0)
end

function fit!(ppm::PartitionedBayesPM, X::AbstractMatrix, y::AbstractMatrix)
    check_regression_data(X, y)
    idx = locate(ppm.P, X)
    for k in unique(idx)
        region_mask = idx .== k
        ppm.scale -= ppm.models[k].lm.scale - ppm.scale0
        fit!(ppm.models[k], X[:, region_mask], y[:, region_mask])
        ppm.scale += ppm.models[k].lm.scale - ppm.scale0
    end
    return ppm.shape += size(X, 2) / 2
end

function lasso_bayespm(
    X::Matrix{Float64},
    y::Matrix{Float64},
    basis::Vector{Index},
    limits::Matrix{Float64},
    λ::Float64,
    λ_intercept::Float64,
    shape0::Float64,
    scale0::Float64,
    Pmax::Int64,
)
    Z = expand(X, basis, limits)
    if length(basis) <= Pmax
        basis_idx = collect(1:length(basis))
    elseif Pmax == 1
        basis_idx = [1]
    else
        local coefs
        try
            coefs = @suppress glmnet(Z[2:end, :]', y[1, :], pmax=Pmax - 1).betas[:, end]
        catch
            path = @suppress glmnet(Z[2:end, :]', y[1, :]).betas
            num_params = [sum(path[:, i] .!= 0) for i in 1:size(path, 2)]
            coefs = path[:, findlast(num_params .<= Pmax - 1)]
        end
        basis_idx = collect(2:length(basis))[coefs.!=0]
        pushfirst!(basis_idx, 1)
    end
    sparse_basis = basis[basis_idx]
    Z = Z[basis_idx, :]
    pm = BayesPM(sparse_basis, limits, λ=λ, shape0=shape0, scale0=scale0)
    pm.lm.Σ[1, 1] = pm.lm.Σ0[1, 1] = λ_intercept^2
    pm.lm.Λ[1, 1] = pm.lm.Λ0[1, 1] = 1 / λ_intercept^2
    if size(Z, 2) > 0
        fit!(pm.lm, Z, y)
    end
    return pm
end

function evidence(models::Vector{BayesPM}, shape0::Float64, scale0::Float64)
    shape, scale, ev = shape0, scale0, 0.0
    for pm in models
        shape += pm.lm.shape - shape0
        scale += pm.lm.scale - scale0
        ev += 0.5 * logdet(pm.lm.Σ) - 0.5 * logdet(pm.lm.Σ0)
    end
    n = 2 * (shape - shape0)
    ev += -(n / 2) * log(2 * π)
    ev += shape0 * log(scale0) - shape * log(scale)
    ev += loggamma(shape) - loggamma(shape0)
    return ev
end

vol(limits::Matrix{Float64}) = prod(limits[:, 2] - limits[:, 1])

function choose_split_polys(
    X::Matrix{Float64},
    y::Matrix{Float64},
    models::Vector{BayesPM},
    k::Int64,
    basis_cache::Vector{Vector{Index}},
    model_cache::Vector{Array{BayesPM,3}},
    space_vol::Float64,
    λ::Float64,
    shape0::Float64,
    scale0::Float64,
    Pmax::Int64,
    min_obs::Vector{Float64},
)
    models_cp = deepcopy(models)
    push!(models_cp, models_cp[1])
    limits = models[k].limits
    num_dims = size(X, 1)
    bc = length(basis_cache)
    best = Dict{String,Any}("ev" => -Inf)
    for d in 1:num_dims
        loc = sum(limits[d, :]) / 2
        mask = X[d, :] .< loc
        left_limits = deepcopy(limits)
        right_limits = deepcopy(limits)
        left_limits[d, 2] = right_limits[d, 1] = loc
        for Jleft in 1:bc, Jright in 1:bc
            if sum(mask) < min_obs[Jleft] || sum(.!mask) < min_obs[Jright]
                continue
            elseif !(isassigned(model_cache[k], 1, d, Jleft) && isassigned(model_cache[k], 2, d, Jright))
                println(sum(mask), "length models = ", length(models_cp))
                model_cache[k][1, d, Jleft] = lasso_bayespm(
                    X[:, mask],
                    y[:, mask],
                    basis_cache[Jleft],
                    left_limits,
                    λ,
                    λ * vol(left_limits) / space_vol,
                    shape0,
                    scale0,
                    Pmax,
                )
                model_cache[k][2, d, Jright] = lasso_bayespm(
                    X[:, .!mask],
                    y[:, .!mask],
                    basis_cache[Jright],
                    right_limits,
                    λ,
                    λ * vol(right_limits) / space_vol,
                    shape0,
                    scale0,
                    Pmax,
                )
            end
            models_cp[k] = model_cache[k][1, d, Jleft]
            models_cp[end] = model_cache[k][2, d, Jright]
            ev = evidence(models_cp, shape0, scale0)
            if ev > best["ev"]
                best["left_pm"] = models[k]
                best["right_pm"] = models[end]
                best["d"] = d
                best["ev"] = ev
                best["mask"] = mask
            end
        end
    end
    return best
end

"""
    PartitionedBayesPM(X::AbstractMatrix, y::AbstractMatrix, limits::Matrix{Float64};
                       <keyword arguments>)

Perform a 1-step look ahead greedy search for a partitioned polynomial model.

# Keyword Arguments
- `Jmax::Int64=3`: The maximum degree of any polynomial model.
- `Jmin::Int64=0`: The minimum degree of any polynomial model.
- `Pmax::Int64=500`: The maximum number of features in a particular regions.
- `Kmax::Int64=200`: The maximum number of regions
- `λ::Float64=1.0`: Prior scaling.
- `shape0::Float64=1e-3`: Inverse-gamma prior shape hyperparameter.
- `scale0::Float64=1e-3`: Inverse-gamma prior scale hyperparameter.
- `ratio::Float64=1.0`: Polynomial degrees are reduced until `size(X, 2) < ratio * length(tpbasis(d, J))`.
- `tol::Float64=1e-4`: The required increase in the model evidence to accept a split.
- `verbose::Bool=true`: Print details of the partition search.
"""
function PartitionedBayesPM(
    X::AbstractMatrix,
    y::AbstractMatrix,
    limits::Matrix{Float64};
    Jmax::Int64=5,
    Pmax::Int64=15,
    Kmax::Int64=200,
    λ::Float64=1.0,
    shape0::Float64=0.01,
    scale0::Float64=0.01,
    ratio::Float64=1.0,
    tol::Float64=1e-3,
    verbose::Bool=true
)
    check_regression_data(X, y)
    check_limits(limits)
    if size(X, 1) != size(limits, 1)
        throw(ArgumentError("X and limits don't match in their first dimension"))
    elseif Jmax < 0
        throw(ArgumentError("must have non-negative Jmax"))
    elseif Kmax <= 0
        throw(ArgumentError("Kmax must be strictly positive"))
    elseif tol < 0.0
        throw(ArgumentError("tolerance must be non-negative"))
    end

    P = Partition(limits)
    idx = ones(Int64, size(X, 2))
    model_cache = [Array{BayesPM,3}(undef, 2, size(X, 1), Jmax + 1)]
    basis_cache = [tpbasis(size(X, 1), J) for J in 0:Jmax]
    space_vol = vol(limits)

    min_obs = [length(b) * ratio for b in basis_cache]
    min_obs[1] = 0
    models = [
        lasso_bayespm(
            X,
            y,
            basis_cache[findlast(size(X, 2) .>= min_obs)],
            limits,
            λ,
            λ,
            shape0,
            scale0,
            Pmax
        )
    ]
    ev = evidence(models, shape0, scale0)

    while length(models) < Kmax
        count = 0
        accepted = false
        K = length(models)
        for k in randperm(K)
            count += 1
            region_mask = idx .== k
            Xk, yk = X[:, region_mask], y[:, region_mask]
            best = choose_split_polys(
                Xk,
                yk,
                models,
                k,
                basis_cache,
                model_cache,
                space_vol,
                λ,
                shape0,
                scale0,
                Pmax,
                min_obs
            )
            if best["ev"] > ev + tol
                models[k] = best["left_pm"]::BayesPM
                push!(models, best["right_pm"]::BayesPM)
                model_cache[k] = Array{BayesPM,3}(undef, 2, size(X, 1), Jmax + 1)
                push!(model_cache, Array{BayesPM,3}(undef, 2, size(X, 1), Jmax + 1))

                # Update the region indices
                region_idx = @view idx[region_mask]
                region_idx[.!best["mask"]::BitVector] .= length(models)

                split!(P, k, best["d"]::Int64)
                ev = best["ev"]::Float64
                break
            end
        end
        if !accepted
            break # Every split has been rejected. Terminate search.
        end
    end
    return PartitionedBayesPM(P, models, shape0, scale0)
end

function (ppm::PartitionedBayesPM)(X::AbstractMatrix)
    idx = locate(ppm.P, X)
    y = zeros(1, length(idx))
    for k in unique(idx)
        region_mask = idx .== k
        y[:, region_mask] = ppm.models[k](X[:, region_mask])
    end
    return y
end