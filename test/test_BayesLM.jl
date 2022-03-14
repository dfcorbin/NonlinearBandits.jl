# Test Sherman-Morrison inverse
A = rand(2, 2)
Ainv = inv(A)
u = rand(2, 1)
B = A + u * u'
@test NonlinearBandits.sherman_morrison_inv(Ainv, u, u) ≈ inv(B)
@test_throws(DimensionMismatch("u and v have mismatched dimensions"),
             NonlinearBandits.sherman_morrison_inv(Ainv, u, zeros(3, 1)))
@test_throws(ArgumentError("u and v should have exactly one column"),
             NonlinearBandits.sherman_morrison_inv(Ainv, zeros(2, 2), zeros(2, 2)))
@test_throws(DimensionMismatch("size of A does not match u/v"),
             NonlinearBandits.sherman_morrison_inv(Ainv, zeros(3, 1), zeros(3, 1)))

## Test Bayesian linear model fitting procedure
d, n, σ = 2, 10000, 2
β = rand(Normal(0, 10), d, 1)
f(x) = β' * x
X, y = NonlinearBandits.gaussian_data(d, n, f; σ=σ)
shape0, scale0 = 0.001, 0.001

lm = BayesLM(d; shape0=shape0, scale0=scale0)
@test_throws(ArgumentError("expected variance is undefined for shape ≤ 1"), std(lm))
fit!(lm, X, y)

prediction_mae = mae(lm(X), mapslices(f, X; dims=1))
β_mae = mae(lm.β, β)
@test β_mae < 0.5
@test prediction_mae < 0.5
@test lm.shape == shape0 + n / 2
@test isapprox(std(lm), σ; atol=0.5)

# Check that sequential updates lead to the same results
sequential_lm = BayesLM(d; shape0=shape0, scale0=scale0)
for i in 1:n
    fit!(sequential_lm, X[:, i:i], y[:, i:i])
end

@test sequential_lm.β ≈ lm.β
@test sequential_lm.Σ ≈ lm.Σ
@test sequential_lm.shape == shape0 + (n / 2)
@test sequential_lm.scale ≈ lm.scale

# Check errors
@test_throws(ArgumentError("d, shape0 and scale0 must be strictly positive"),
             BayesLM(0; shape0=0.0, scale0=0.0))
@test_throws(ArgumentError("y should have exactly 1 row"), fit!(lm, X, y'))
@test_throws(DimensionMismatch("X and y don't match in dimension 2"),
             fit!(lm, X[:, 1:2], y))
@test_throws(DimensionMismatch("X does not have the same number of predictors as the model"),
             fit!(lm, X[1:1, :], y))
@test_throws(ArgumentError("X does not have the same number of predictors as the model"),
             lm(X[1:1, :]))
