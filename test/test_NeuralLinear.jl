a = [1, 2]
A = NonlinearBandits.action_matrix(a, 2)
@test A == [1.0 0.0; 0.0 1.0]

d = 5
num_actions = 2
layer_sizes = [64, 64]
encoder = NeuralEncoder(d, num_actions, layer_sizes)
X = rand(5, 2)
@test size(encoder(X)) == (layer_sizes[end], 2)

y = rand(1, 2)
Ypred = cpu(encoder.nn(gpu(X)))
ls = (y[1, 1] - Ypred[a[1], 1])^2 + (y[1, 2] - Ypred[a[2], 2])^2
ls /= 2
@test encoder.loss(gpu(X), gpu(A), gpu(y)) ≈ ls

# Create bandit trajectory
d, n = 2, 5000
mf = (x -> 3 * x[1] + 5 * x[2], x -> 4 * x[1])
X = rand(d, n)
a = rand(1:2, n)
r = zeros(1, n)
for i in 1:n
    r[1, i] = mf[a[i]](X[:, i]) + rand(Normal(0, 0.1))
end

enc = NeuralEncoder(d, length(mf), [32, 32])
fit!(enc, X, a, r, 10; verbose=false)
Ypred = cpu(enc.nn(gpu(X)))
@test mae(Ypred[1:1, :], mapslices(mf[1], X; dims=1)) < 0.1
@test mae(Ypred[2:2, :], mapslices(mf[2], X; dims=1)) < 0.1

# Test updating NeuralLinear policy appears to work
limits = repeat([-1.0 1.0], d, 1)
csampler = UniformContexts(limits)
rsampler = GaussianRewards(mf)
metrics = (FunctionalRegret(mf),)

# Test NeuralLinear appears to be learning the optimal policy.
batch_size = 1000
num_batches = 3
initial_batches = retrain_freq = 1
nl = NeuralLinear(d, length(mf), layer_sizes, initial_batches, [1:num_batches...], 20)
metrics = (FunctionalRegret(mf),)
driver = StandardDriver(csampler, nl, rsampler, metrics)
run!(num_batches, batch_size, driver)

regret = metrics[1].regret[(end - batch_size + 1):end]
@test sum(regret .== 0.0) / length(regret) >= 0.95