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
- `??::Float64=1.0`: Prior scaling.
- `shape0::Float64=1e-3`: Inverse-gamma prior shape hyperparameter.
- `scale0::Float64=1e-3`: Inverse-gamma prior scale hyperparameter.
"""
function PartitionedBayesPM(
    P::Partition,
    Js::Vector{Int64};
    ??::Float64=1.0,
    shape0::Float64=1e-3,
    scale0::Float64=1e-3,
)
    if length(P.regions) != length(Js)
        throw(ArgumentError("must supply a value of J for every region in P"))
    end
    d = size(P.limits, 1)
    basis = [tpbasis(d, J) for J in Js]
    models = [
        BayesPM(basis[i], P.regions[i]; ??=??, shape0=shape0, scale0=scale0) for
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

"""
    lasso_selection(X::AbstractMatrix, y::AbstractMatrix, Pmax::Int64, intercept::Bool)

Choose the first `Pmax` features introduced by a LASSO solution path.

# Arguments

- `X::AbstractMatrix`: A matrix with observations stored as columns.
- `y::AbstractMatrix`: A matrix with 1 row of response variables. 
- `Pmax::Int64`: The maximum number of predictors.
- `intercept::Bool`: `true` if the first row of `X` are the intercept features
"""
function lasso_selection(X::AbstractMatrix, y::AbstractMatrix, Pmax::Int64, intercept::Bool)
    if size(X, 1) <= Pmax
        return [1:size(X, 1)...]
    end
    X, Pmax = intercept ? (X[2:end, :], Pmax - 1) : (X, Pmax)
    ?? = @suppress glmnet(X', y[1, :]; pmax=Pmax).betas[:, end]
    indices = intercept ? [1] : Int64[]
    @inbounds for j in 1:length(??)
        if ??[j] != 0.0
            i = intercept ? j + 1 : j
            push!(indices, i)
        end
    end
    return indices
end

function evidence(models::Vector{BayesPM}, shape0::Float64, scale0::Float64)
    shape, scale, ev = shape0, scale0, 0.0
    for pm in models
        shape += pm.lm.shape - shape0
        scale += pm.lm.scale - scale0
        ev += 0.5 * logdet(pm.lm.??) - 0.5 * logdet(pm.lm.??0)
    end
    n = 2 * (shape - shape0)
    ev += -(n / 2) * log(2 * ??)
    ev += shape0 * log(scale0) - shape * log(scale)
    ev += loggamma(shape) - loggamma(shape0)
    return ev
end

function _conditional_degree_selection!(
    X::AbstractMatrix, # Data to train new polynomial
    y::AbstractMatrix, # Data to train new polynomial
    k::Int64, # Subregion to be replaced
    d::Int64, # Dimension index of model cache
    lateral::Int64, # Lateral index of model cache
    sub_limits::Matrix{Float64}, # Limits of the new polynomial
    Jmax::Int64,
    Pmax::Int64,
    models::Vector{BayesPM},
    model_cache::Vector{Array{BayesPM,3}},
    basis_cache::Vector{Vector{Index}},
    ??::Float64,
    shape0::Float64,
    scale0::Float64,
    ratio::Float64,
)
    n = size(X, 2)
    best_ev = -Inf
    best_pm::Union{Nothing,BayesPM} = nothing
    models_cp = deepcopy(models)
    for J in 0:Jmax
        if J > 0 && n < ratio * length(basis_cache[J + 1])
            continue
        end
        if !isassigned(model_cache[k], d, lateral, J + 1)
            # Polynomial not found in cache, fit a new degree J polynmomial
            # and store it.
            basis = basis_cache[J + 1]
            Z = expand(X, basis, sub_limits; J=J)
            idx = lasso_selection(Z, y, Pmax, true)
            models_cp[k] = BayesPM(
                basis[idx], sub_limits; ??=??, shape0=shape0, scale0=scale0
            )
            if n > 0
                fit!(models_cp[k].lm, Z[idx, :], y)
            end
            model_cache[k][d, lateral, J + 1] = models_cp[k]
        else
            # Polynomial found in cache.
            models_cp[k] = model_cache[k][d, lateral, J + 1]
        end
        ev = evidence(models_cp, shape0, scale0)
        if ev > best_ev
            best_ev = ev
            best_pm = models_cp[k]
        end
    end
    return best_pm
end

vol(limits::Matrix{Float64}) = prod(limits[:, 2] - limits[:, 1])

"""
    PartitionedBayesPM(X::AbstractMatrix, y::AbstractMatrix, limits::Matrix{Float64};
                       <keyword arguments>)

Perform a 1-step look ahead greedy search for a partitioned polynomial model.

# Keyword Arguments
- `Jmax::Int64=3`: The maximum degree of any polynomial model.
- `Pmax::Int64=500`: The maximum number of features in a particular regions.
- `Kmax::Int64=200`: The maximum number of regions
- `??::Float64=1.0`: Prior scaling.
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
    Jmax::Int64=3,
    Pmax::Int64=500,
    Kmax::Int64=200,
    ??::Float64=1.0,
    shape0::Float64=1e-3,
    scale0::Float64=1e-3,
    ratio::Float64=1.0,
    tol::Float64=1e-4,
    verbose::Bool=true,
)
    check_regression_data(X, y)
    check_limits(limits)
    if size(X, 1) != size(limits, 1)
        throw(ArgumentError("X and limits don't match in their first dimension"))
    elseif Jmax < 0
        throw(ArgumentError("Jmax must be non-negative"))
    elseif Kmax <= 0
        throw(ArgumentError("Kmax must be strictly positive"))
    elseif tol < 0.0
        throw(ArgumentError("tolerance must be non-negative"))
    end

    P = Partition(limits)
    idx = ones(Int64, size(X, 2))
    model_cache = [Array{BayesPM,3}(undef, size(X, 1), 2, Jmax + 1)]
    basis_cache = [tpbasis(size(X, 1), J) for J in 0:Jmax]
    models = Vector{BayesPM}(undef, 1)
    space_vol = vol(limits)

    # Set up the intial polynomial for the full space and clear the cache.
    # We don't need to cache the inital entry as we have already accepted it.
    # The cache indices supplied to _conditional_degree_selection! are just
    # placeholders
    models[1] = _conditional_degree_selection!(
        X,
        y,
        1,
        1,
        1,
        limits,
        Jmax,
        Pmax,
        models,
        model_cache,
        basis_cache,
        ??,
        shape0,
        scale0,
        ratio,
    )
    model_cache[1] = Array{BayesPM,3}(undef, size(X, 1), 2, Jmax + 1)
    ev = evidence(models, shape0, scale0)

    while length(models) < Kmax
        count = 0
        accepted = false
        K = length(models)
        for k in randperm(K)
            # Iterate over every subregion and test to see if any split
            # improves the model evidence beyond the tolerance.

            count += 1
            best = Dict{String,Any}("ev" => ev + tol)
            models_cp = deepcopy(models)
            push!(models_cp, models_cp[1])
            region_mask = idx .== k
            Xk, yk = X[:, region_mask], y[:, region_mask]
            for d in 1:size(X, 1)
                # Region k is split in every dimension and two polynomials are fitted
                # to the resulting subregions. The best details of the best dimensional
                # split are stored in the dictionary `best`

                if verbose
                    msg = "\rRegion = $count/$K; Dimension = $d/$(size(X, 1)); Evidence = $ev"
                    print(msg)
                end
                loc = sum(P.regions[k][d, :]) / 2
                left_lims, right_lims = deepcopy(P.regions[k]), deepcopy(P.regions[k])
                left_lims[d, 2], right_lims[d, 1] = loc, loc
                left_region_mask = Xk[d, :] .< loc
                left_pm = _conditional_degree_selection!(
                    Xk[:, left_region_mask],
                    yk[:, left_region_mask],
                    k,
                    d,
                    1,
                    left_lims,
                    Jmax,
                    Pmax,
                    models,
                    model_cache,
                    basis_cache,
                    ?? * vol(left_lims) / space_vol,
                    shape0,
                    scale0,
                    ratio,
                )
                right_pm = _conditional_degree_selection!(
                    Xk[:, .!left_region_mask],
                    yk[:, .!left_region_mask],
                    k,
                    d,
                    2,
                    right_lims,
                    Jmax,
                    Pmax,
                    models,
                    model_cache,
                    basis_cache,
                    ?? * vol(right_lims) / space_vol,
                    shape0,
                    scale0,
                    ratio,
                )
                models_cp[k] = left_pm
                models_cp[end] = right_pm
                tmp_ev = evidence(models_cp, shape0, scale0)
                if tmp_ev > best["ev"] >= ev + tol
                    accepted = true
                    best["ev"] = tmp_ev
                    best["d"] = d
                    best["left_region_mask"] = left_region_mask
                    best["left_pm"], best["right_pm"] = left_pm, right_pm
                end
            end
            if accepted
                # Update models and reset cache for k'th region. Add extra 
                # cache entry for the additional region.
                models[k] = best["left_pm"]::BayesPM
                push!(models, best["right_pm"]::BayesPM)
                model_cache[k] = Array{BayesPM,3}(undef, size(X, 1), 2, Jmax + 1)
                push!(model_cache, Array{BayesPM,3}(undef, size(X, 1), 2, Jmax + 1))

                # Update the region indices
                region_idx = @view idx[region_mask]
                region_idx[.!best["left_region_mask"]::BitVector] .= length(models)

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