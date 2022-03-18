mf = (x -> 100, x -> -100)
met = FunctionalRegret(mf)
@test met(rand(1, 2), [1, 2], rand(1, 1)) == [0.0, 200.0]
