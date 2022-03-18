d, batch_size = 5, 100
limits = repeat([-1.0 1.0], d, 1)
mf = (x -> 100, x -> -100)
csampler = UniformContexts(limits)
policy = RandomPolicy(length(mf))
rsampler = GaussianRewards(mf)
metrics = (FunctionalRegret(mf),)
driver = StandardDriver(csampler, policy, rsampler, metrics)

X, a, r = driver(batch_size)
@test size(X) == (d, batch_size)
@test all(r[1, a .== 1] .> 90)
@test all(r[1, a .== 2] .< -90)
@test all(metrics[1].regret[a .== 1] .== 0.0)
@test all(metrics[1].regret[a .== 2] .== 200.0)

driver = StandardDriver(csampler, policy, rsampler)
X, a, r = driver(batch_size)
@test size(X) == (d, batch_size)
@test all(r[1, a .== 1] .> 90)
@test all(r[1, a .== 2] .< -90)

@test_throws(
    ArgumentError("num_batches and batch_size must be positive"), run!(0, 0, driver)
)
