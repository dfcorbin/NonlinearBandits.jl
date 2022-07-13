function test_legendre()
    basis = tensor_product_basis(2, 2)
    expected_basis = [
        Index(Int64[], Int64[]),
        Index([1], [1]),
        Index([1], [2]),
        Index([2], [1]),
        Index([2, 1], [1, 1]),
        Index([2], [2]),
    ]
    basis_comparison = [
        (idx1.dims == idx2.dims) && (idx2.degrees == idx2.degrees)
        for (idx1, idx2) in zip(basis, expected_basis)
    ]
    @test all(basis_comparison)

    basis = [Index(Int64[], Int64[]), Index([1, 2], [2, 1])]
    X = [0.1 0.2]
    limits = [-1.0 1.0; -1.0 1.0]
    Z = expand(X, limits, basis)

    polys = [x -> 1.0, x -> x, x -> 1.5 * x^2 - 0.5]
    expected_Z = [1.0 polys[3](X[1, 1]) * polys[2](X[1, 2])]
    @test expected_Z ≈ Z
end


function test_PolyModel()
    sd = 0.1
    d = 2
    basis = tensor_product_basis(d, 3)
    coefs = rand(Normal(0, 1), length(basis))
    limits = repeat([0.0 1.0], d, 1)

    function f(x)
        X = reshape(x, (1, :))
        Z = expand(X, limits, basis)
        return (Z*coefs)[1]
    end

    n = 5000
    X, y = gaussian_data(n, d, f, sd=sd)
    pm = PolyModel(limits, basis)
    fit!(pm, X, y)

    estimated_coefs = get_coefs(pm)
    @test isapprox(coefs, estimated_coefs, atol=0.1)
    @test isapprox(sd^2, variance(pm), atol=0.005)
    @test get_shape(pm) == 0.01 + 0.5 * n
    @test pm(X[1:1, :]) ≈ expand(X[1:1, :], limits, basis) * estimated_coefs
end


test_legendre()
test_PolyModel()