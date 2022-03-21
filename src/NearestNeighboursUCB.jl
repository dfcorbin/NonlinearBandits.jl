mutable struct NearestNeighboursUCB <: AbstractPolicy
    t::Int64
    batch::Int64
    num_arms::Int64
    initial_batches::Int64
    data::BanditDataset
    Xa::Vector{Matrix{Float64}}
    ra::Vector{Matrix{Float64}}
    θ::Float64
    φ::Float64

    function NearestNeighboursUCB(
        d::Int64, num_arms::Int64, initial_batches::Int64; θ::Float64=1.0, φ::Float64=1.0
    )
        Xa = [Matrix{Float64}(undef, d, 0) for _ in 1:num_arms]
        ra = [Matrix{Float64}(undef, 1, 0) for _ in 1:num_arms]
        data = BanditDataset(d)
        return new(0, 0, num_arms, initial_batches, data, Xa, ra, θ, φ)
    end
end

function nearest_neighbours(x::AbstractVector, X::AbstractMatrix)
    if size(X, 1) != length(x)
        throw(DimensionMismatch("length of x should match size(X, 1)"))
    end
    n = size(X, 2)
    distances = zeros(n)
    @inbounds for i in 1:n
        distances[i] = sqrt(sum((X[:, i] .- x) .^ 2))
    end
    idx = sortperm(distances)
    dst = distances[idx]
    return idx, dst
end

function _choose_k(x::AbstractVector, X::AbstractMatrix, t::Int64, θ::Float64, φ::Float64)
    idx, dst = nearest_neighbours(x, X)
    U = zeros(length(idx))
    @inbounds for k in 1:length(idx)
        radius = dst[k]
        U[k] = sqrt(θ * log(t) / k) + φ * log(t) * radius
    end
    min_k = argmin(U)
    return idx[1:min_k], U[min_k]
end

function (pol::NearestNeighboursUCB)(X::AbstractMatrix)
    n = size(X, 2)
    actions = zeros(Int64, n)
    if pol.batch <= pol.initial_batches
        for i in 1:n
            actions[i] = (pol.t + i) % pol.num_arms + 1
        end
        return actions
    end

    ucb = zeros(pol.num_arms)
    for i in 1:n
        for a in 1:(pol.num_arms)
            idx, U = _choose_k(X[:, i], pol.Xa[a], pol.t + 1, pol.θ, pol.φ)
            ucb[a] = sum(pol.ra[a][idx]) / length(idx) + U
        end
        actions[i] = argmax(ucb)
    end
    return actions
end

function update!(
    pol::NearestNeighboursUCB,
    X::AbstractMatrix,
    a::AbstractVector{<:Int},
    r::AbstractMatrix,
)
    pol.t += size(X, 2)
    pol.batch += 1
    append_data!(pol.data, X, a, r)
    for i in unique(a)
        pol.Xa[i] = hcat(pol.Xa[i], X[:, a .== i])
        pol.ra[i] = hcat(pol.ra[i], r[:, a .== i])
    end
end