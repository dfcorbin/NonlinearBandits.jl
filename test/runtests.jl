using NonlinearBandits
using Test
using Distributions: Uniform, Normal
import Base: ==

function mae(x1, x2)
    return sum(abs.(x1 - x2)) / length(x1)
end

@testset "BayesLM.jl" begin
    include("test_BayesLM.jl")
end

@testset "BayesPM.jl" begin
    include("test_BayesPM.jl")
end

@testset "PartitionedBayesPM.jl" begin
    include("test_PartitionedBayesPM.jl")
end

@testset "bandits.jl" begin
    include("test_bandits.jl")
end

@testset "PolynomialThompsonSampling." begin
    include("test_PolynomialThompsonSampling.jl")
end