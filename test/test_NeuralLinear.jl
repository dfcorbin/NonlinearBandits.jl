function test_output_matrix()
    a = [1, 2]
    A = NonlinearBandits.output_matrix(a, 2)
    @test A == [1.0 0.0; 0.0 1.0]
end


function test_NeuralEncoder()
    num_inputs = 1
    num_outputs = 2
    layer_sizes = [32, 32]
    encoder = NeuralEncoder(num_inputs, num_outputs, layer_sizes)
    n = 2
    X = rand(n, num_inputs)
    Z = encoder(X)
    @test size(Z) == (n, layer_sizes[end])

    # Test the loss is computed correctly
    Y = predict(encoder, X)
    X1 = X' |> gpu
    a = [1, 2]
    A = NonlinearBandits.output_matrix(a, num_outputs) |> gpu
    y = rand(1, 2) |> gpu
    loss = float(encoder.loss(X1, A, y))
    y = cpu(y)
    @test loss ≈ ((Y[a[1], 1] - y[1])^2 + (Y[a[2], 2] - y[2])^2) / 2

    # Create some synthetic data
    n = 5000
    mean_funs = (x -> 5 * sin(2 * π * x[1]), x -> 5 * cos(2 * π * x[1]))
    X = rand(n, num_inputs)
    actions = rand(1:length(mean_funs), n)
    rewards =
        [mean_funs[a](x) + rand(Normal(0, 0.1)) for (a, x) in zip(actions, eachrow(X))]
    fit!(encoder, X, actions, rewards, 50, verbose = false)

    x = rand(1, 1)
    @test isapprox(
        predict(encoder, x),
        [mean_funs[1](x[1, 1]) mean_funs[2](x[1, 1])],
        atol = 0.5,
    )
end

function test_NeuralLinear()
    num_inputs = 1
    num_arms = 2
    nl = NeuralLinear(num_inputs, num_arms, [32, 32], 20, [500 * i for i = 1:100], 50)

    limits = repeat([-1.0 1.0], num_inputs, 1)
    mf = (x -> x[1]^3, x -> x[1])
    csampler = UniformContexts(limits)
    rsampler = GaussianRewards(mf, 0.1)
    metrics = (FunctionalRegret(mf),)
    driver = StandardDriver(csampler, nl, rsampler, metrics)
    run!(5000, 1, driver, verbose = false)
    @test sum(metrics[1].regret[(end-100):end] .== 0.0) > 95
end


test_output_matrix()
test_NeuralEncoder()
test_NeuralLinear()