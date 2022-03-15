# The code in this section has not been optimised, it has been kept simple for testing
# purposes. I will implement a more optimised version if there is ever a need for it.

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

function _conditional_degree_selection(
    X::Matrix{Float64}, # Data to train new polynomial
    y::Matrix{Float64}, # Data to train new polynomial
    k::Int64, # Subregion to be replaced
    d::Int64,
    side::Int64, # Left subregion or right (used for cache location)
    sub_limits::Matrix{Float64},
    Jmax::Int64,
    models::Vector{BayesPM},
    model_cache::Vector{Array{BayesPM,3}},
    basis_cache::Vector{Vector{Index}},
    λ::Float64,
    shape0::Float64,
    scale0::Float64,
)
    ev = fill(-Inf, Jmax + 1)
    pms = Vector{BayesPM}(undef, Jmax + 1)
    for J in 0:Jmax
        # No models are added to the cache when computing the intial model
        # across the space (since there is no concept of left or right). This
        # is accounted for by checking for an empty dictionary.
        models_cp = deepcopy(models)
        is_cache_empty = isempty(model_cache)
        if is_cache_empty || !isassigned(model_cache[k], d, side, J + 1)
            basis = basis_cache[J + 1]
            models_cp[k] = BayesPM(basis, sub_limits; λ=λ, shape0=shape0, scale0=scale0)
            fit!(models_cp[k], X, y)
            if !is_cache_empty
                model_cache[k][d, side, J + 1] = models_cp[k]
            end
        else
            models_cp[k] = model_cache[k][d, side, J + 1]
        end
        pms[J + 1] = models_cp[k]
        ev[J + 1] = evidence(models_cp, shape0, scale0)
    end
    return pms[argmax(ev)]
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
)
    model_cache = Vector{Array{BayesPM,3}}(undef, 0)
    basis_cache = [tpbasis(size(X, 1), J) for J in 0:Jmax]
    P = Partition(limits)
    idx = ones(Int64, size(X, 2))
    models = Vector{BayesPM}(undef, 1)
    models[1] = _conditional_degree_selection(
        X, y, 1, 1, 1, limits, Jmax, models, model_cache, basis_cache, λ, shape0, scale0
    )
    push!(model_cache, Array{BayesPM,3}(undef, size(X, 1), 2, Jmax + 1))
    ev = evidence(models, shape0, scale0)

    while length(models) < Kmax
        K = length(models)
        accepted = false
        count = 0
        for k in randperm(K)
            count += 1
            best = Dict{String,Any}("ev" => ev + tol)
            region_idx = idx .== k
            Xs, ys = X[:, region_idx], y[:, region_idx]
            for d in 1:size(X, 1)
                print("\rRegion = $count/$K; Dimension = $d/$(size(X, 1)); Evidence = $ev")
                loc = sum(P.regions[k][d, :]) / 2
                left_lims, right_lims = deepcopy(P.regions[k]), deepcopy(P.regions[k])
                left_lims[d, 2], right_lims[d, 1] = loc, loc
                left = Xs[d, :] .< loc
                left_pm = _conditional_degree_selection(
                    Xs[:, left],
                    ys[:, left],
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
                right_pm = _conditional_degree_selection(
                    Xs[:, .!left],
                    ys[:, .!left],
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
                models_cp = deepcopy(models)
                models_cp[k] = left_pm
                push!(models_cp, right_pm)
                tmp_ev = evidence(models_cp, shape0, scale0)
                if tmp_ev > best["ev"] >= ev + tol
                    accepted = true
                    best["ev"] = tmp_ev
                    best["d"] = d
                    best["left"] = left
                    best["left_pm"], best["right_pm"] = left_pm, right_pm
                end
            end
            if accepted
                ev = best["ev"]::Float64
                models[k] = best["left_pm"]::BayesPM
                push!(models, best["right_pm"]::BayesPM)
                model_cache[k] = Array{BayesPM,3}(undef, size(X, 1), 2, Jmax + 1)
                push!(model_cache, Array{BayesPM,3}(undef, size(X, 1), 2, Jmax + 1))
                split!(P, k, best["d"]::Int64)
                region_idx[best["left"]::BitVector] .= 0
                idx[region_idx] .= length(models)
                break
            end
        end
        if !accepted
            break
        end
    end
    return models
end