mutable struct PolynomialThompsonSampling <: AbstractPolicy
    t::Int64
    steps::Int64
    data::BanditDataset
    arms::Vector{PartitionedBayesPM}
    initial_steps::Int64
    α::Float64
    retrain_freq::Int64
    limits::Matrix{Float64}

    Jmax::Int64
    Pmax::Int64
    Kmax::Int64
    λ::Float64
    shape0::Float64
    scale0::Float64
    ratio::Float64
    tol::Float64
    verbose_retrain::Bool

    function PolynomialThompsonSampling(
        limits::Matrix{Float64},
        num_arms::Int64,
        initial_steps::Int64,
        retrain_freq::Int64;
        α::Float64=1.0,
        Jmax::Int64=3,
        Pmax::Int64=100,
        Kmax::Int64=500,
        λ::Float64=1.0,
        shape0::Float64=1e-3,
        scale0::Float64=1e-3,
        ratio::Float64=1.0,
        tol::Float64=1e-4,
        verbose_retrain::Bool=false,
    )
        d = size(limits, 1)
        data = BanditDataset(d)
        arms = Vector{PartitionedBayesPM}(undef, num_arms)
        return new(
            0,
            0,
            data,
            arms,
            initial_steps,
            α,
            retrain_freq,
            limits,
            Jmax,
            Pmax,
            Kmax,
            λ,
            shape0,
            scale0,
            ratio,
            tol,
            verbose_retrain,
        )
    end
end

function (pol::PolynomialThompsonSampling)(X::AbstractMatrix)
    n = size(X, 2)
    num_arms = length(pol.arms)
    actions = zeros(Int64, n)

    # Check if inital steps have been completed
    if pol.steps <= pol.initial_steps
        for i in 1:n
            actions[i] = (pol.t + i) % num_arms + 1
        end
        return actions
    end

    thompson_samples = zeros(num_arms)
    for i in 1:n
        for a in 1:num_arms
            shape, scale = shape_scale(pol.arms[a])
            x = X[:, i:i]
            k = locate(pol.arms[a].P, x)[1]
            varsim = rand(InverseGamma(shape, scale))
            β = pol.arms[a].models[k].lm.β
            Σ = pol.α * varsim * pol.arms[a].models[k].lm.Σ
            βsim = rand(MvNormal(β[:, 1], Σ))
            z = expand(
                x,
                pol.arms[a].models[k].basis,
                pol.arms[a].P.regions[k];
                J=pol.arms[a].models[k].J,
            )
            thompson_samples[a] = βsim' * z[:, 1]
        end
        actions[i] = argmax(thompson_samples)
    end
    return actions
end

function update!(
    pol::PolynomialThompsonSampling,
    X::AbstractMatrix,
    a::AbstractVector{Int64},
    r::AbstractMatrix,
)
    pol.t += size(X, 2)
    pol.steps += 1
    add_data!(pol.data, X, a, r)
    if pol.steps < pol.initial_steps
        return nothing
    end
    if pol.steps % pol.retrain_freq == 0 || pol.steps == pol.initial_steps
        for a in 1:length(pol.arms)
            Xa, ra = arm_data(pol.data, a)
            pol.arms[a] = PartitionedBayesPM(
                Xa,
                ra,
                pol.limits;
                Jmax=pol.Jmax,
                Pmax=pol.Pmax,
                Kmax=pol.Kmax,
                λ=pol.λ,
                shape0=pol.shape0,
                scale0=pol.scale0,
                ratio=pol.ratio,
                tol=pol.tol,
                verbose=pol.verbose_retrain,
            )
        end
    else
        for i in unique(a)
            fit!(pol.arms[i], X[:, a .== i], r[:, a .== i])
        end
    end
end
