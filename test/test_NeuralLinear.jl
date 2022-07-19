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
    rewards = [mean_funs[a](x) + rand(Normal(0, 1)) for (a, x) in zip(actions, eachrow(X))]
    fit!(encoder, X, actions, rewards, 50, verbose = false)

    x = rand(1, 1)
    @test isapprox(
        predict(encoder, x),
        [mean_funs[1](x[1, 1]) mean_funs[2](x[1, 1])],
        atol = 0.3,
    )
end


test_output_matrix()
test_NeuralEncoder()


# # # Create bandit trajectory
# # d, n = 2, 5000
# # mf = (x -> 3 * x[1] + 5 * x[2], x -> 4 * x[1])
# # X = rand(d, n)
# # a = rand(1:2, n)
# # r = zeros(1, n)
# # for i in 1:n
# #     r[1, i] = mf[a[i]](X[:, i]) + rand(Normal(0, 0.1))
# # end

# # enc = NeuralEncoder(d, length(mf), [32, 32])
# # fit!(enc, X, a, r, 10; verbose=false)
# # Ypred = cpu(enc.nn(gpu(X)))
# # @test mae(Ypred[1:1, :], mapslices(mf[1], X; dims=1)) < 0.1
# # @test mae(Ypred[2:2, :], mapslices(mf[2], X; dims=1)) < 0.1

# # # Test updating NeuralLinear policy appears to work
# # limits = repeat([-1.0 1.0], d, 1)
# # csampler = UniformContexts(limits)
# # rsampler = GaussianRewards(mf)
# # metrics = (FunctionalRegret(mf),)

# # # Test NeuralLinear appears to be learning the optimal policy.
# # batch_size = 1000
# # num_batches = 3
# # initial_batches = retrain_freq = 1
# # nl = NeuralLinear(d, length(mf), layer_sizes, initial_batches, [1:num_batches...], 20)
# # metrics = (FunctionalRegret(mf),)
# # driver = StandardDriver(csampler, nl, rsampler, metrics)
# # run!(num_batches, batch_size, driver)

# # regret = metrics[1].regret[(end - batch_size + 1):end]
# # @test sum(regret .== 0.0) / length(regret) >= 0.95