function ==(i1::Index, i2::Index)
    return i1.dim == i2.dim && i1.deg == i2.deg
end

polys = [x -> 1.0, x -> x, x -> (3 * x^2 - 1) / 2, x -> (5 * x^3 - 3 * x) / 2]

# Test legendre recurrance relation
@test NonlinearBandits.legendre_next(0.5, 2, 0.5, 1.0) == polys[3](0.5)
@test_throws(DomainError("x lies outside [- 1 - 1e-4, 1 + 1e-4]"),
             NonlinearBandits.legendre_next(1.1, 0, 0.0, 0.0))
@test_throws(ArgumentError("tol must be non-negative"),
             NonlinearBandits.legendre_next(0.0, 2, 0.0, 0.0; tol=-1.0))

# Test tensor product basis construction
@test_throws(ArgumentError("dim contains repeated values"), Index([1, 1], [1, 1]))
@test_throws(ArgumentError("dim contains values < 1"), Index([0, 1], [1, 1]))
@test_throws(ArgumentError("deg contains values < 0"), Index([1, 2], [-1, 1]))
@test_throws(DimensionMismatch("dim and deg have mismatched length"),
             Index([1, 2], [1]))
@test tpbasis(2, 2) ==
      Index[Index(Int64[], Int64[]), Index([1], [1]), Index([1], [2]),
            Index([2], [1]), Index([2, 1], [1, 1]), Index([2], [2])]

d = 2
basis = Index[Index([1], [2]), Index([1, 2], [1, 3])]
limits = repeat([-1.0 1.0], d, 1)
X = repeat(rand(d, 1), 1, 2)
p = repeat([polys[2 + 1](X[1, 1]); polys[1 + 1](X[1, 1]) * polys[3 + 1](X[2, 1]);;], 1, 2)
@test expand(X, basis, limits) ≈ p
@test_throws(ArgumentError("invalid expansion limits, expected size (2, 2)"),
             expand(X, basis, [-1 1]))
@test_throws(ArgumentError("limits[:, 1] must be <= limits[:, 2]"),
             expand(X, basis, [-1 1; 1 -1]),)
@test_throws(ArgumentError("J should be >= 0"), expand(X, basis, limits; J=-1))

# Test polynomial regression
d, n, J = 2, 50000, 3
basis = tpbasis(d, J)
nbas = length(basis)
β = rand(Normal(0, 10), nbas, 1)

function f(x)
    z = reshape(x, (:, 1))
    z = expand(z, basis, limits; J=J)
    return (β' * z)[1, 1]
end

X, y = NonlinearBandits.gaussian_data(d, n, f; σ=2)
pm = BayesPM(basis, limits; λ=200.0)
fit!(pm, X, y)

@test mae(pm.lm.β, β) < 1.0
@test pm.lm.shape == 1e-3 + n / 2
@test isapprox(std(pm), 2; atol=0.5)