mutable struct NearestNeighboursUCB <: AbstractPolicy
    t::Int64
    batch::Int64
    initial_batches::Int64
    Xs::Vector{Matrix{Float64}}
    rs::Vector{Vector{Float64}}
    theta::Float64
    varphi::Float64

    function NearestNeighboursUCB(
        d::Int64,
        num_arms::Int64,
        initial_batches::Int64;
        theta::Float64 = 1.0,
        varphi::Float64 = 1.0,
    )
        Xs = [Matrix{Float64}(undef, 0, d) for _ = 1:num_arms]
        rs = [Float64[] for _ = 1:num_arms]
        return new(0, 0, initial_batches, Xs, rs, theta, varphi)
    end
end


function compute_distances(x::Vector{Float64}, X::Matrix{Float64})
    distances = [sqrt(sum((x .- x1) .^ 2)) for x1 in eachrow(X)]
    order = sortperm(distances)
    distances = distances[order]
    return order, distances
end


function compute_uncertainty(
    x::Vector{Float64},
    X::Matrix{Float64},
    rewards::Vector{Float64},
    t::Int64,
    theta::Float64,
    varphi::Float64,
)
    order, distances = compute_distances(x, X)
    uncertainties = Vector{Float64}(undef, length(order))
    for k = 1:length(order)
        uncertainties[k] = sqrt(theta * log(t) / k) + varphi * log(t) * distances[k]
    end
    uncertainty, num_neighbours = findmin(uncertainties)
    return uncertainty, order[1:num_neighbours]
end


function (pol::NearestNeighboursUCB)(X::Matrix{Float64})
    n = size(X, 1)
    actions = Vector{Int64}(undef, n)
    if pol.batch <= pol.initial_batches
        for i = 1:n
            actions[i] = (pol.t + i) % length(pol.Xs) + 1
        end
        return actions
    end

    ucb = Vector{Float64}(undef, length(pol.Xs))
    for i = 1:n
        for a = 1:length(pol.Xs)
            uncertainty, neighbours = compute_uncertainty(
                X[i, :],
                pol.Xs[a],
                pol.rs[a],
                pol.t + 1,
                pol.theta,
                pol.varphi,
            )
            ucb[a] = sum(pol.rs[a][neighbours]) / length(neighbours) + uncertainty
        end
        actions[i] = argmax(ucb)
    end
    return actions
end


function update!(
    pol::NearestNeighboursUCB,
    X::Matrix{Float64},
    actions::Vector{Int64},
    rewards::Vector{Float64},
)
    pol.t += size(X, 1)
    pol.batch += 1
    for a in unique(actions)
        pol.Xs[a] = vcat(pol.Xs[a], X[actions.==a, :])
        pol.rs[a] = vcat(pol.rs[a], rewards[actions.==a])
    end
end