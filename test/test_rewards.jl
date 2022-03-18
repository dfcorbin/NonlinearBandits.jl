mf = (x -> 100.0, x -> -100.0)
rsampler = GaussianRewards(mf)
@test rsampler(rand(1, 1), [1])[1, 1] > 90
@test rsampler(rand(1, 1), [2])[1, 1] < -90
