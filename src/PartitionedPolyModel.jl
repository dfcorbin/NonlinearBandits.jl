mutable struct Region
    key::Int64
    limits::Matrix{Float64}
    d::Int64
    loc::Float64
    left::Region
    right::Region

    function Region(key::Int64, limits::Matrix{Float64})
        return new(key, limits)
    end
end


mutable struct Partition
    space::Region
    regions::Vector{Region}
end


function Partition(limits::Matrix{Float64})
    space = Region(1, limits)
    return Partition(space, [space])
end


function split!(prt::Partition, idx::Int64, d::Int64)
    region = prt.regions[idx]
    loc = sum(region.limits[d, :]) * 0.5
    left_limits = deepcopy(region.limits)
    right_limits = deepcopy(region.limits)
    left_limits[d, 2] = right_limits[d, 1] = loc

    region.key = 0
    region.d = d
    region.loc = loc
    region.left = Region(idx, left_limits)
    region.right = Region(length(prt.regions) + 1, right_limits)

    prt.regions[idx] = region.left
    push!(prt.regions, region.right)
    return nothing
end


function locate(region::Region, x::AbstractVector)
    if region.key != 0
        return region.key
    end
    if x[region.d] < region.loc
        locate(region.left, x)
    else
        locate(region.right, x)
    end
end


function locate(prt::Partition, X::Matrix{Float64})
    return [locate(prt.space, x) for x in eachrow(X)]
end


function _sparse_polymodel(
    X::Matrix{Float64},
    y::Vector{Float64},
    limits::Matrix{Float64},
    basis::Vector{Index},
    max_param::Int64,
    prior_shape::Float64,
    prior_scale::Float64,
    regularization::Float64,
    intercept_regularization::Float64 = regularization,
)
    Z = expand(X, limits, basis)
    if length(basis) <= max_param
        basis_idx = collect(1:length(basis))
    elseif max_param == 1 # Assumes the first basis function is degree 0
        basis_idx = [1]
    else
        local coefs
        try
            coefs = @suppress glmnet(Z[:, 2:end], y, pmax = max_param - 1).betas[:, end]
        catch
            path = @suppress glmnet(Z[:, 2:end]', y).betas
            num_params = [sum(path[:, i] .!= 0) for i = 1:size(path, 2)]
            coefs = path[:, findlast(num_params .<= max_param - 1)]
        end
        basis_idx = collect(2:length(basis))[coefs.!=0]
        pushfirst!(basis_idx, 1)
    end
    basis = basis[basis_idx]
    Z = Z[:, basis_idx]
    pm = PolyModel(
        limits,
        basis,
        prior_shape = prior_shape,
        prior_scale = prior_scale,
        regularization = regularization,
    )
    pm.lm.post.prec[1, 1] = pm.lm.prior.prec[1, 1] = intercept_regularization
    if length(y) > 0
        fit!(pm.lm, Z, y)
    end
    return pm
end


function evidence(models::Vector{PolyModel}, prior_shape::Float64, prior_scale::Float64)
    shape, scale, ev = prior_scale, prior_shape, 0.0
    for pm in models
        shape += get_shape(pm) - prior_shape
        scale += get_scale(pm) - prior_scale
        ev += 0.5 * logdet(get_prior_prec(pm)) - 0.5 * logdet(get_prec(pm))
    end
    n = 2 * (shape - prior_shape)
    ev += -(n / 2) * log(2 * π)
    ev += prior_shape * log(prior_scale) - shape * log(scale)
    ev += loggamma(shape) - loggamma(prior_shape)
    return ev
end


function _maximise_evidence!(
    X::Matrix{Float64},
    y::Vector{Float64},
    k::Int64,
    models::Vector{PolyModel},
    max_degree::Int64,
    min_obs::Vector{Float64},
    model_cache::Vector{Array{PolyModel,3}},
    basis_cache::Vector{Vector{Index}},
    max_param::Int64,
    prior_shape::Float64,
    prior_scale::Float64,
    regularization::Float64,
)
    num_dims = size(X, 2)
    limits = models[k].limits
    models_cp = deepcopy(models)
    push!(models_cp, models_cp[1])
    best = Dict{String,Any}("evidence" => -Inf)
    for d = 1:num_dims
        loc = sum(limits[d, :]) * 0.5
        left_limits = deepcopy(limits)
        right_limits = deepcopy(limits)
        left_limits[d, 2] = right_limits[d, 1] = loc
        mask = X[:, d] .< loc
        Xleft, yleft = X[mask, :], y[mask]
        Xright, yright = X[.!mask, :], y[.!mask]
        for jl = 0:max_degree, jr = 0:max_degree
            if length(yleft) < min_obs[jl+1] || length(yright) < min_obs[jr+1]
                continue
            elseif !(
                isassigned(model_cache[k], 1, d, jl + 1) &&
                isassigned(model_cache[k], 2, d, jr + 1)
            )
                model_cache[k][1, d, jl+1] = _sparse_polymodel(
                    Xleft,
                    yleft,
                    left_limits,
                    basis_cache[jl+1],
                    max_param,
                    prior_shape,
                    prior_scale,
                    regularization * 2,
                    regularization,
                )
                model_cache[k][2, d, jr+1] = _sparse_polymodel(
                    Xright,
                    yright,
                    right_limits,
                    basis_cache[jr+1],
                    max_param,
                    prior_shape,
                    prior_scale,
                    regularization * 2,
                    regularization,
                )
            end
            models_cp[k] = model_cache[k][1, d, jl+1]
            models_cp[end] = model_cache[k][2, d, jr+1]
            ev = evidence(models_cp, prior_shape, prior_scale)
            # println("jl = $jl, jr = $jr, d = $d, ev = $ev")
            if ev > best["evidence"]
                best["evidence"] = ev
                best["left_pm"] = models_cp[k]
                best["right_pm"] = models_cp[end]
                best["d"] = d
            end
        end
    end
    return best
end