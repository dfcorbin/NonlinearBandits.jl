function test_LinearModel()
    n, d = 5000, 2
    coefs = rand(Normal(0, 1), d)
    f(x) = coefs' * x
    sd = 0.1
    X, y = gaussian_data(n, d, f, sd=sd)
    lm = LinearModel(d)
    fit!(lm, X, y)
    estimated_coefs = get_coefs(lm)
    @test isapprox(coefs, estimated_coefs, atol=0.1)
    @test isapprox(sd^2, variance(lm), atol=0.005)
    @test get_shape(lm) == 0.01 + 0.5 * n
    @test lm(X[1:1, :]) ≈ X[1:1, :] * estimated_coefs
end


test_LinearModel()