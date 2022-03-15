mutable struct Partition
    limits::Matrix{Float64}
    regions::Vector{Matrix{Float64}}
end

"""
    Partition(limits::Matrix{Float64})

Construct an object capable of storing a hyperrectangular partition.
"""
function Partition(limits::Matrix{Float64})
    check_limits(limits)
    return Partition(deepcopy(limits), deepcopy([limits]))
end

"""
    split!(P::Partition, k::Int64, d::Int64) 

Split the `k`'th subregion of `P` into equal halves in dimension `d`.
"""
function split!(P::Partition, k::Int64, d::Int64)
    right = deepcopy(P.regions[k])
    loc = sum(right[d, :]) / 2
    P.regions[k][d, 2] = loc
    right[d, 1] = loc
    return push!(P.regions, right)
end

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

mutable struct PartitionedBayesPM
    P::Partition
    models::Vector{BayesPM}
    shape0::Float64
    scale0::Float64
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

function _conditional_degree_selection!(
    X::Matrix{Float64}, # Data to train new polynomial
    y::Matrix{Float64}, # Data to train new polynomial
    k::Int64, # Subregion to be replaced
    d::Int64, # Dimension index of model cache
    lateral::Int64, # Lateral index of model cache
    sub_limits::Matrix{Float64}, # Limits of the new polynomial
    Jmax::Int64,
    models::Vector{BayesPM},
    model_cache::Vector{Array{BayesPM,3}},
    basis_cache::Vector{Vector{Index}},
    λ::Float64,
    shape0::Float64,
    scale0::Float64,
)
    best_ev = -Inf
    best_pm::Union{Nothing, BayesPM} = nothing
    models_cp = deepcopy(models)
    for J in 0:Jmax
        if !isassigned(model_cache[k], d, lateral, J + 1)
            # Polynomial not found in cache, fit a new degree J polynmomial
            # and store it.
            basis = basis_cache[J + 1]
            models_cp[k] = BayesPM(basis, sub_limits; λ=λ, shape0=shape0, scale0=scale0)
            fit!(models_cp[k], X, y)
            model_cache[k][d, lateral, J + 1] = models_cp[k]
        else
            # Polynomial found in cache.
            models_cp[k] = model_cache[k][d, lateral, J + 1]
        end
        ev = evidence(models_cp, shape0, scale0)
        if ev > best_ev
            best_pm = models_cp[k]
        end
    end
    return best_pm
end

function auto_partitioned_bayes_pm(
    X::Matrix{Float64},
    y::Matrix{Float64},
    limits::Matrix{Float64};
    Jmax::Int64=3,
    Kmax::Int64=200,
    λ::Float64=1.0,
    shape0::Float64=1e-3,
    scale0::Float64=1e-3,
    tol::Float64=1e-4,
    verbose::Bool=true,
)
    P = Partition(limits)
    idx = ones(Int64, size(X, 2))
    model_cache = [Array{BayesPM,3}(undef, size(X, 1), 2, Jmax + 1)]
    basis_cache = [tpbasis(size(X, 1), J) for J in 0:Jmax]
    models = Vector{BayesPM}(undef, 1)

    # Set up the intial polynomial for the full space and clear the cache.
    # We don't need to cache the inital entry as we have already accepted it.
    # The cache indices supplied to _conditional_degree_selection! are just
    # placeholders
    models[1] = _conditional_degree_selection!(
        X, y, 1, 1, 1, limits, Jmax, models, model_cache, basis_cache, λ, shape0, scale0
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
                    models,
                    model_cache,
                    basis_cache,
                    λ,
                    shape0,
                    scale0,
                )
                right_pm = _conditional_degree_selection!(
                    Xk[:, .!left_region_mask],
                    yk[:, .!left_region_mask],
                    k,
                    d,
                    2,
                    right_lims,
                    Jmax,
                    models,
                    model_cache,
                    basis_cache,
                    λ,
                    shape0,
                    scale0,
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
                region_mask[best["left_region_mask"]::BitVector] .= 0
                idx[region_mask] .= length(models)

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