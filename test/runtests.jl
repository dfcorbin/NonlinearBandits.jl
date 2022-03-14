using NonlinearBandits
using Test
using Distributions: Uniform, Normal

function mae(x1, x2)
    return sum(abs.(x1 - x2)) / length(x1)
end

@testset "BayesLM.jl" begin
    include("test_BayesLM.jl")
end

@testset "BayesPM.jl" begin
    include("test_BayesPM.jl")
end