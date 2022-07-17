function test_standard_driver()
    d, batch_size = 5, 100
    limits = repeat([-1.0 1.0], d, 1)
    mf = (x -> 100, x -> -100)
    csampler = UniformContexts(limits)
    policy = UniformPolicy(length(mf))
    rsampler = GaussianRewards(mf, 1.0)
    metrics = AbstractMetric[FunctionalRegret(mf)]
    driver = StandardDriver(csampler, policy, rsampler, metrics)

    X, actions, rewards = driver(batch_size)
    @test size(X) == (batch_size, d)
    @test all(rewards[actions.==1] .> 90)
    @test all(rewards[actions.==2] .< -90)
    @test all(metrics[1].regret[actions.==1] .== 0.0)
    @test all(metrics[1].regret[actions.==2] .== 200.0)

    driver = StandardDriver(csampler, policy, rsampler)
    X, actions, rewards = driver(batch_size)
    @test size(X) == (batch_size, d)
    @test all(rewards[actions.==1] .> 90)
    @test all(rewards[actions.==2] .< -90)
end


test_standard_driver()


# FIX TEST ERRORS
# GO BACK TO USING VARARG TUPLES

# # Test latent driver
# tform = z -> 100 + z[1]
# driver = LatentDriver(csampler, policy, rsampler, tform)
# X, a, r = driver(1)
# @test 99 <= X[1, 1] <= 101