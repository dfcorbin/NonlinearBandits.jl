# Test uniform contexts
d, n = 5, 10
csampler = UniformContexts(repeat([10.0 11.0], d, 1))
X = csampler(n)
@test size(X) == (d, n)
@test all(10 .<= X .<= 11)