using Test, NonlinearBandits
d = 5
n = 10
A = 5
β = rand(d, A)
mf = (x -> 100.0, x -> -100.0)

contexts = UniformContexts(repeat([10.0 11.0], d, 1))
rewards = GaussianRewards(mf; σ=1.0)

X = contexts(n)

@test size(X) == (d, n)
@test all(10 .<= X .<= 11)
@test all(90 .<= rewards(X, ones(Int64, n)) .<= 110)
@test all(-110 .<= rewards(X, 2 .* ones(Int64, n)) .<= 110)