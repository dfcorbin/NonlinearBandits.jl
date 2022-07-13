function test_legendre()
    polys = legendre.(0:3)
    @test (polys[1].degrees, polys[1].coefs) == ([0], [1.0])
    @test (polys[2].degrees, polys[2].coefs) == ([1], [1.0])
    @test (polys[3].degrees, polys[3].coefs) == ([2, 0], [1.5, -0.5])
end


test_legendre()