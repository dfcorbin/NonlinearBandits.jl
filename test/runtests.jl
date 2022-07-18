import Base: ==
using Distributions: Normal
using NonlinearBandits
using Test


function gaussian_data(n, d, f; sd = 1.0)
    X = rand(n, d)
    y = [f(x) + rand(Normal(0, sd)) for x in eachrow(X)]
    return X, y
end


@testset "LinearModel" begin
    include("test_LinearModel.jl")
end


@testset "PolyModel" begin
    include("test_PolyModel.jl")
end


@testset "PartitionedPolyModel" begin
    include("test_PartitionedPolyModel.jl")
end


@testset "Contexts" begin
    include("test_contexts.jl")
end


@testset "Rewards" begin
    include("test_rewards.jl")
end


@testset "Metrics" begin
    include("test_metrics.jl")
end


@testset "Drivers" begin
    include("test_drivers.jl")
end


@testset "PolynomialThompsonSampling" begin
    include("test_PolynomialThompsonSampling.jl")
end