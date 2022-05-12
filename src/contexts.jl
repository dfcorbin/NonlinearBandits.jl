"""
    UniformContexts(limits::Matrix{Float64})

Construct a callable object to generate uniform contexts.
"""
struct UniformContexts <: AbstractContextSampler
    limits::Matrix{Float64}

    function UniformContexts(limits::Matrix{Float64})
        check_limits(limits)
        return new(limits)
    end
end

function (sampler::UniformContexts)(n::Int64)
    d = size(sampler.limits, 1)
    X = zeros(d, n)
    for i in 1:d
        X[i, :] = rand(Uniform(sampler.limits[i, 1], sampler.limits[i, 2]), n)
    end
    return X
end

struct WheelContexts <: AbstractContextSampler end

function (sampler::WheelContexts)(n::Int64)
    radius = sqrt.(rand(n))
    angle = rand(n) * 2 * π
    X = zeros(2, n)
    for i in 1:n
        X[1, i] = radius[i] * cos(angle[i])
        X[2, i] = radius[i] * sin(angle[i])
    end
    return X
end
