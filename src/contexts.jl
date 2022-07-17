abstract type AbstractContextSampler end


struct UniformContexts <: AbstractContextSampler
    limits::Matrix{Float64}
end


function (sampler::UniformContexts)(n::Int64)
    d = size(sampler.limits, 1)
    X = Matrix{Float64}(undef, n, d)
    for j = 1:d
        unif = Uniform(sampler.limits[j, 1], sampler.limits[j, 2])
        X[:, j] = rand(unif, n)
    end
    return X
end


struct WheelContexts <: AbstractContextSampler end


function (sampler::WheelContexts)(n::Int64)
    radius = sqrt.(rand(n))
    angle = rand(n) * 2 * π
    X = zeros(n, 2)
    for i = 1:n
        X[i, 1] = radius[i] * cos(angle[i])
        X[i, 2] = radius[i] * sin(angle[i])
    end
    return X
end