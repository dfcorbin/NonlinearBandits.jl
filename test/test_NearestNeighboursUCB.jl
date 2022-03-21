using NonlinearBandits, Test

X = [3 1 2]
x = [1]
@test nearest_neighbours(x, X) == ([2, 3, 1], [0.0, 1.0, 2.0])
