abstract type AbstractMetric end


mutable struct FunctionalRegret{T<:Tuple{Vararg{<:Function}}} <: AbstractMetric
    mean_funs::T
    regret::Vector{Float64}

    function FunctionalRegret(mean_funs::Tuple{Vararg{<:Function}})
        return new{typeof(mean_funs)}(mean_funs, Float64[])
    end
end


function (met::FunctionalRegret)(
    X::Matrix{Float64},
    actions::Vector{Int64},
    rewards::Vector{Float64},
)
    regret = Vector{Float64}(undef, size(X, 1))
    for (i, (x, a)) in enumerate(zip(eachrow(X), actions))
        means = [mf(x) for mf in met.mean_funs]
        regret[i] = maximum(means) - means[a]
    end
    met.regret = vcat(met.regret, regret)
end


mutable struct WheelRegret <: AbstractMetric
    radius::Float64
    means::Tuple{Float64,Float64,Float64}
    regret::Vector{Float64}

    function WheelRegret(radius::Float64, means::Tuple{Float64,Float64,Float64})
        return new(radius, means, Float64[])
    end
end


function (met::WheelRegret)(
    X::Matrix{Float64},
    actions::Vector{Int64},
    rewards::Vector{Float64},
)
    regret = Vector{Float64}(undef, size(X, 1))
    for (i, (x, a)) in enumerate(zip(eachrow(X), actions))
        dst = sqrt(sum(x .^ 2))
        if dst <= met.radius
            regret[i] = a == 1 ? 0.0 : met.means[1] - met.means[2]
        elseif a == 1
            regret[i] = met.means[3] - met.means[1]
        elseif (actions[i] == 2 && x[1] >= 0 && x[2] >= 0) ||
               (actions[i] == 3 && x[1] >= 0 && x[2] < 0) ||
               (actions[i] == 4 && x[1] < 0 && x[2] >= 0) ||
               (actions[i] == 5 && x[1] < 0 && x[2] < 0)
            regret[i] = 0.0
        else
            regret[i] = met.means[3] - met.means[2]
        end
    end
    met.regret = vcat(met.regret, regret)
end