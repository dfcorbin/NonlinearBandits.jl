using NonlinearBandits
using Test
using CUDA
using Distributions: Uniform, Normal
using Flux: gpu, cpu
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

@testset "contexts.jl" begin
    include("test_contexts.jl")
end

@testset "rewards.jl" begin
    include("test_rewards.jl")
end

@testset "metrics.jl" begin
    include("test_metrics.jl")
end

@testset "drivers.jl" begin
    include("test_drivers.jl")
end

@testset "PolynomialThompsonSampling.jl" begin
    include("test_PolynomialThompsonSampling.jl")
end

@testset "NeuralLinear.jl" begin
    include("test_NeuralLinear.jl")
end

@testset "NearestNeighboursUCB.jl" begin
    include("test_NearestNeighboursUCB.jl")
end