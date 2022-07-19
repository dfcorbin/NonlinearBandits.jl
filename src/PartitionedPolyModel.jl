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


function locate(prt::Partition, X::AbstractMatrix)
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
    space_vol::Float64,
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
                    regularization * space_vol / vol(left_limits),
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
                    regularization * space_vol / vol(right_limits),
                    regularization,
                )
            end
            models_cp[k] = model_cache[k][1, d, jl+1]
            models_cp[end] = model_cache[k][2, d, jr+1]
            ev = evidence(models_cp, prior_shape, prior_scale)
            if ev > best["evidence"]
                best["evidence"] = ev
                best["left_pm"] = models_cp[k]
                best["right_pm"] = models_cp[end]
                best["mask"] = mask
                best["d"] = d
            end
        end
    end
    return best
end


function vol(limits::Matrix{Float64})
    return prod(limits[:, 2] .- limits[:, 1])
end


mutable struct PartitionedPolyModel
    polys::Vector{PolyModel}
    prt::Partition
    prior_shape::Float64
    prior_scale::Float64
end


function PartitionedPolyModel(
    X::Matrix{Float64},
    y::Vector{Float64},
    limits::Matrix{Float64};
    max_degree::Int64 = 5,
    max_param::Int64 = 15,
    max_models::Int64 = 200,
    min_data::Int64 = 2,
    data_constraint::Float64 = 1.0,
    prior_shape::Float64 = 0.01,
    prior_scale::Float64 = 0.01,
    regularization::Float64 = 1.0,
    tolerance::Float64 = 1e-3,
    verbose::Bool = true,
)
    # Initial setup
    n, d = size(X)
    basis_cache = [tensor_product_basis(d, deg) for deg = 0:max_degree]
    min_obs = [max(min_data, length(b) * data_constraint) for b in basis_cache]
    models = [
        _sparse_polymodel(
            X,
            y,
            limits,
            basis_cache[findlast(n .>= min_obs)],
            max_param,
            prior_shape,
            prior_scale,
            regularization,
            regularization,
        ),
    ]
    ev = evidence(models, prior_shape, prior_scale)
    model_cache = [Array{PolyModel,3}(undef, 2, d, max_degree + 1)]
    locations = ones(Int64, n)
    space_vol = vol(limits)
    prt = Partition(limits)

    while length(models) < max_models
        accepted = false
        for k in randperm(length(models))
            mask = locations .== k
            best = _maximise_evidence!(
                X[mask, :],
                y[mask],
                k,
                models,
                max_degree,
                min_obs,
                model_cache,
                basis_cache,
                max_param,
                prior_shape,
                prior_scale,
                regularization,
                space_vol,
            )
            if best["evidence"] > ev + tolerance
                ev = best["evidence"]
                models[k] = best["left_pm"]
                push!(models, best["right_pm"])
                split!(prt, k, best["d"])
                region_locations = @view locations[mask]
                region_locations[.!best["mask"]] .= length(models)
                model_cache[k] = Array{PolyModel,3}(undef, 2, d, max_degree + 1)
                push!(model_cache, Array{PolyModel,3}(undef, 2, d, max_degree + 1))
                accepted = true
            end
        end
        if !accepted
            break
        end
    end
    return PartitionedPolyModel(models, prt, prior_shape, prior_scale)
end


function (ppm::PartitionedPolyModel)(X::Matrix{Float64})
    locations = locate(ppm.prt, X)
    y = zeros(size(X, 1))
    for k in unique(locations)
        mask = locations .== k
        y[mask] = ppm.polys[k](X[mask, :])
    end
    return y
end


function fit!(ppm::PartitionedPolyModel, X::Matrix{Float64}, y::Vector{Float64})
    locations = locate(ppm.prt, X)
    for k in unique(locations)
        mask = locations .== k
        fit!(ppm.polys[k], X[mask, :], y[mask])
    end
    return nothing
end


function get_shape(ppm::PartitionedPolyModel)
    shape = ppm.prior_shape
    for pm in ppm.polys
        shape += get_shape(pm) - ppm.prior_shape
    end
    return shape
end


function get_scale(ppm::PartitionedPolyModel)
    scale = ppm.prior_scale
    for pm in ppm.polys
        scale += get_scale(pm) - ppm.prior_scale
    end
    return scale
end


function get_coefs(ppm::PartitionedPolyModel, k::Int64)
    return get_coefs(ppm.polys[k])
end


function get_prec(ppm::PartitionedPolyModel, k::Int64)
    return get_prec(ppm.polys[k])
end


function get_partition(ppm::PartitionedPolyModel)
    return ppm.prt
end


function get_region_limits(ppm::PartitionedPolyModel, k::Int64)
    return ppm.polys[k].limits
end


function get_basis(ppm::PartitionedPolyModel, k::Int64)
    return get_basis(ppm.polys[k])
end


function get_degree(ppm::PartitionedPolyModel, k::Int64)
    return get_degree(ppm.polys[k])
end


function posterior_sample(
    ppm::PartitionedPolyModel,
    x::AbstractVector,
    inflation::Float64 = 1.0,
)
    x1 = reshape(x, (1, :))
    k = locate(ppm.prt, x1)[1]
    return posterior_sample(ppm.polys[k], x, inflation)
end