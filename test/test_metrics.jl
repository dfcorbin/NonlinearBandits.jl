function test_functional_regret()
    mf = (x -> 100, x -> -100)
    met = FunctionalRegret(mf)
    met(rand(2, 1), [1, 2], rand(1))
    @test met.regret == [0.0, 200.0]
end


function test_wheel_regret()
    X = [
        -0.1 0.1
        0.1 0.1
        0.1 -0.1
        -0.1 -0.1
    ]
    expected = [
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
    ]
    for a = 1:5
        met = WheelRegret(0.8, (10.0, 0.0, 20.0))
        actions = fill(a, 4)
        r = zeros(4)
        met(X, actions, r)
        @test met.regret == expected[a]
    end

    X = [
        -1.0 1.0
        1.0 1.0
        1.0 -1.0
        -1.0 -1.0
    ]
    expected = [
        [10.0, 10.0, 10.0, 10.0],
        [20.0, 0.0, 20.0, 20.0],
        [20.0, 20.0, 0.0, 20.0],
        [0.0, 20.0, 20.0, 20.0],
        [20.0, 20.0, 20.0, 0.0],
    ]
    for a = 1:5
        met = WheelRegret(0.8, (10.0, 0.0, 20.0))
        actions = fill(a, 4)
        r = zeros(4)
        met(X, actions, r)
        @test met.regret == expected[a]
    end
end


test_functional_regret()
test_wheel_regret()