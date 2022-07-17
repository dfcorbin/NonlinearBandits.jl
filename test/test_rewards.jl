function test_gaussian_rewards()
    mf = (x -> 100.0, x -> -100.0)
    rsampler = GaussianRewards(mf, 0.1)
    @test rsampler(rand(1, 1), [1])[1] > 90
    @test rsampler(rand(1, 1), [2])[1] < -90
end


function test_wheel_rewards()
    sampler = WheelRewards(0.8, (10.0, 0.0, 20.0), 0.01)
    X = [
        -0.1 0.1
        0.1 0.1
        0.1 -0.1
        -0.1 -0.1
    ]

    # Inside radius
    @test isapprox(sampler(X, [1, 1, 1, 1]), [10.0, 10.0, 10.0, 10.0], atol = 0.1)
    @test isapprox(sampler(X, [2, 2, 2, 2]), [0.0, 0.0, 0.0, 0.0], atol = 0.1)
    @test isapprox(sampler(X, [3, 3, 3, 3]), [0.0, 0.0, 0.0, 0.0], atol = 0.1)
    @test isapprox(sampler(X, [4, 4, 4, 4]), [0.0, 0.0, 0.0, 0.0], atol = 0.1)
    @test isapprox(sampler(X, [5, 5, 5, 5]), [0.0, 0.0, 0.0, 0.0], atol = 0.1)

    X = [
        -1.0 1.0
        1.0 1.0
        1.0 -1.0
        -1.0 -1.0
    ]

    @test isapprox(sampler(X, [1, 1, 1, 1]), [10.0, 10.0, 10.0, 10.0], atol = 0.1)
    @test isapprox(sampler(X, [2, 2, 2, 2]), [0.0, 20.0, 0.0, 0.0], atol = 0.1)
    @test isapprox(sampler(X, [3, 3, 3, 3]), [0.0, 0.0, 20.0, 0.0], atol = 0.1)
    @test isapprox(sampler(X, [4, 4, 4, 4]), [20.0, 0.0, 0.0, 0.0], atol = 0.1)
    @test isapprox(sampler(X, [5, 5, 5, 5]), [0.0, 0.0, 0.0, 20.0], atol = 0.1)
end


test_gaussian_rewards()
test_wheel_rewards()