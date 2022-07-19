function test_nearest_neighbours()
    X = [3.0; 1.0; 2.0;;]
    x = [1.0]
    @test NonlinearBandits.compute_distances(x, X) == ([2, 3, 1], [0.0, 1.0, 2.0])
end


function test_NearestNeighboursUCB()
    d = 1
    limits = [-1.0 1.0]
    csampler = UniformContexts(limits)
    mean_funs = (x -> x[1], x -> -x[1])
    rsampler = GaussianRewards(mean_funs, 0.1)
    policy = NearestNeighboursUCB(d, length(mean_funs), 5)
    metrics = (FunctionalRegret(mean_funs),)
    driver = StandardDriver(csampler, policy, rsampler, metrics)
    num_steps = 10000
    run!(num_steps, 1, driver, verbose = false)
    @test sum(metrics[1].regret[(num_steps-1000):end] .== 0.0) > 900
end


test_nearest_neighbours()
test_NearestNeighboursUCB()