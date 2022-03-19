"""
    PolynomialThompsonSampling(limits::Matrix{Float64}, num_arms::Int64, 
                               initial_batches::Int64, retrain_freq::Int64;
                               <keyword arguments>)

Construct a Thompson sampling policy that uses a [`PartitionedBayesPM`](@ref) to
model the expected rewards.

# Arguments

- `limits::Matrix{Float64}`: Matrix with two columns defining the lower/upper limits of the space.
- `num_arms::Int64`: The number of available actions.
- `inital_batches::Int64`: The number of batches to sample before training the polnomial
    models.
- `retrain_freq::Int64`: The frequency (in terms of batches) at which the partition/basis
    selection is retrained from scratch.
- `α::Float64=1.0`: Thompson sampling inflation. `α > 1` and increasing alpha increases the
    amount of exploration.
- `Jmax::Int64=3`: The maximum degree of any polynomial region.
- `Pmax::Int64=100`: The maximum number of features in any polynomial region.
- `Kmax::Int64=500`: The maximum number of regions in the partition.
- `λ::Float64=1.0`: Prior scaling.
- `shape0::Float64=1e-3`: Inverse-gamma prior shape hyperparameter.
- `scale0::Float64=1e-3`: Inverse-gamma prior scale hyperparameter.
- `ratio::Float64=1.0`: Polynomial degrees are reduced until `size(X, 2) < ratio * length(tpbasis(d, J))`.
- `tol::Float64=1e-4`: The required increase in the model evidence to accept a split.
- `verbose_retrain::Bool=true`: Print details of the partition search.
"""
mutable struct PolynomialThompsonSampling <: AbstractPolicy
    t::Int64
    batches::Int64
    data::BanditDataset
    arms::Vector{PartitionedBayesPM}
    initial_batches::Int64
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
        initial_batches::Int64,
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
            initial_batches,
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

    # Check if inital batches have been completed
    if pol.batches <= pol.initial_batches
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
    a::AbstractVector{<:Int},
    r::AbstractMatrix,
)
    pol.t += size(X, 2)
    pol.batches += 1
    append_data!(pol.data, X, a, r)
    if pol.batches < pol.initial_batches
        return nothing
    end
    if pol.batches % pol.retrain_freq == 0 || pol.batches == pol.initial_batches
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
