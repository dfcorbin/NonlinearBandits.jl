function test_uniform_contexts()
    d, n = 5, 10
    csampler = UniformContexts(repeat([10.0 11.0], d, 1))
    X = csampler(n)
    @test size(X) == (n, d)
    @test all(10 .<= X .<= 11)
end


function test_wheel_contexts()
    csampler = WheelContexts()
    n = 20000
    X = csampler(n)
    dst = [sqrt(sum(x .^ 2)) for x in eachrow(X)]
    r = 0.5
    expected = r^2 * n
    tol = 0.1 * expected
    @test expected - tol <= sum(dst .<= 0.5) <= expected + tol
end


test_uniform_contexts()
test_wheel_contexts()
