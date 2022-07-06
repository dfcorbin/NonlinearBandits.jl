mf = (x -> 100, x -> -100)
met = FunctionalRegret(mf)
met(rand(1, 2), [1, 2], rand(1, 1))
@test met.regret == [0.0, 200.0]
