function test_truncate_batch()
    limits = [0.0 1.0; -1.0 0.0]
    truncated = NonlinearBandits.truncate_batch(limits, [-1.0 -0.5; 0.5 0.5])
    @test truncated == [0.0 -0.5; 0.5 0.0]
end


function test_PolynomialThompsonSampling()
    # Going to set up a well specified problem and see if Thompson Sampling
    # Converges to an optimal policy
    d = 2
    limits = [-1.0 1.0; -1.0 1.0]
    degree = 3
    basis = tensor_product_basis(d, degree)
    fun_coefs = [rand(Normal(0, 1), length(basis)) for _ = 1:2]

    function mf_maker(i)
        function f(x)
            x = reshape(x, (1, :))
            z = expand(x, limits, basis)[1, :]
            return fun_coefs[i]' * z
        end
        return f
    end

    mean_funs = Tuple([mf_maker(_) for _ = 1:2])
    csampler = UniformContexts(limits)
    rsampler = GaussianRewards(mean_funs, 0.1)
    policy = PolynomialThompsonSampling(
        d,
        length(mean_funs),
        5,
        [2^i for i = 1:20];
        regularization = 1.0,
        max_param = 1000,
    )
    metrics = (FunctionalRegret(mean_funs),)
    driver = StandardDriver(csampler, policy, rsampler, metrics)
    num_batches, batch_size = 30000, 1
    run!(num_batches, batch_size, driver; verbose = false)
    @test sum(metrics[1].regret[(num_batches-1000):end] .== 0.0) > 980
end


test_truncate_batch()
test_PolynomialThompsonSampling()