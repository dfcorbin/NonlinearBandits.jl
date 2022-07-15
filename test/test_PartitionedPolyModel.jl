function test_Partition()
    limits = [-1.0 1.0; -1.0 1.0]
    prt = Partition(limits)
    split!(prt, 1, 1)
    split!(prt, 2, 2)

    @test prt.regions[1].limits == [-1.0 0.0; -1.0 1.0]
    @test prt.regions[1].key == 1
    @test prt.regions[2].limits == [0.0 1.0; -1.0 0.0]
    @test prt.regions[2].key == 2
    @test prt.regions[3].limits == [0.0 1.0; 0.0 1.0]
    @test prt.regions[3].key == 3

    @test prt.space.left.limits == [-1.0 0.0; -1.0 1.0]
    @test prt.space.right.left.limits == [0.0 1.0; -1.0 0.0]
    @test prt.space.right.right.limits == [0.0 1.0; 0.0 1.0]

    X = [-1.0 1.0; 0.0 -1.0; 1.0 1.0]
    @test locate(prt, X) == [1, 2, 3]
end


function ==(i1::Index, i2::Index)
    if i1.dims == i2.dims && i1.degrees == i2.degrees
        return true
    else
        return false
    end
end


function test_sparse_polymodel()
    d = 2
    basis = tensor_product_basis(d, 3)
    sparse_basis = basis[[1, 3, 5]]
    coefs = rand(Normal(0, 1), length(sparse_basis))
    limits = [0.0 1.0; 0.0 1.0]

    function f(x)
        x = reshape(x, (1, :))
        Z = expand(x, limits, sparse_basis)
        return (Z*coefs)[1]
    end

    n = 5000
    X, y = gaussian_data(n, d, f, sd = 0.1)
    pm = NonlinearBandits._sparse_polymodel(X, y, limits, basis, 3, 0.01, 0.01, 1.0)
    @test length(pm.basis) == 3
    @test all([b in sparse_basis for b in pm.basis])

    pm = NonlinearBandits._sparse_polymodel(X, y, limits, basis, 1, 0.01, 0.01, 1.0)
    @test pm.basis == basis[1:1]

    pm = NonlinearBandits._sparse_polymodel(X, y, limits, basis, 100, 0.01, 0.01, 1.0)
    @test pm.basis == basis
end


function test_maximise_evidence()
    d = 2
    degrees = [2, 5]
    bases = [tensor_product_basis(d, deg) for deg in degrees]
    coefs = [rand(Normal(0, 1), length(b)) for b in bases]
    limits = [0.0 1.0; 0.0 1.0]
    left_limits = [0.0 1.0; 0.0 0.5]
    right_limits = [0.0 1.0; 0.5 1.0]

    function f(x)
        x = reshape(x, (1, :))
        if x[1, 2] < 0.5
            Z = expand(x, left_limits, bases[1])
            return (Z*coefs[1])[1]
        else
            Z = expand(x, right_limits, bases[2])
            return (Z*coefs[2])[1]
        end
    end

    n = 5000
    X, y = gaussian_data(n, d, f, sd = 0.1)
    max_degree = 6
    models = [
        NonlinearBandits._sparse_polymodel(
            X,
            y,
            limits,
            tensor_product_basis(d, max_degree),
            1000,
            0.01,
            0.01,
            1.0,
        ),
    ]
    model_cache = [Array{PolyModel,3}(undef, 2, d, max_degree + 1)]
    basis_cache = [tensor_product_basis(d, deg) for deg = 0:max_degree]
    min_obs = Float64[length(b) for b in basis_cache]
    min_obs[1] = 0
    best = NonlinearBandits._maximise_evidence!(
        X,
        y,
        1,
        models,
        max_degree,
        min_obs,
        model_cache,
        basis_cache,
        1000,
        0.01,
        0.01,
        1.0,
    )
    @test best["d"] == 2
    @test best["left_pm"].basis == bases[1]
    @test best["right_pm"].basis == bases[2]
end


test_Partition()
test_sparse_polymodel()
test_maximise_evidence()