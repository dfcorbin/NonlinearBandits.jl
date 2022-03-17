using NonlinearBandits

d = 1
limits = repeat([-1.0 1.0], d, 1)
mf = (x -> 5 * x[1], x -> 10 * sin(5^(2 * x[1]) * x[1]))
num_arms = length(mf)

csampler = UniformContexts(limits)
rsampler = GaussianRewards(mf)
policy = PolynomialThompsonSampling(limits, num_arms, 1, 10)
driver = StandardDriver(csampler, policy, rsampler)
run!(100, 1000, driver)