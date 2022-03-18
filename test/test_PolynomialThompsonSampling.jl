using NonlinearBandits, Test

d = 2
limits = repeat([-1.0 1.0], d, 1)
mf = (x -> 10.0, x -> -10.0)
num_arms = length(mf)

csampler = UniformContexts(limits)
rsampler = GaussianRewards(mf)
inital_batches = 5
policy = PolynomialThompsonSampling(limits, num_arms, inital_batches, 5; λ=10.0)
driver = StandardDriver(csampler, policy, rsampler)
batch_size = 100

run!(inital_batches, batch_size, driver)
X1, y1 = arm_data(policy.data, 1)
X2, y2 = arm_data(policy.data, 2)
@test size(X1, 2) + size(X2, 2) == inital_batches * batch_size
@test policy.arms[1].models[1].J == 0
@test abs(policy.arms[1].models[1].lm.β[1] - 10.0) < 0.1
@test policy.arms[2].models[2].J == 0
@test abs(policy.arms[2].models[1].lm.β[1] + 10.0) < 0.1