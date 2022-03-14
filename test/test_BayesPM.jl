polys = [x -> 1.0, x -> x, x -> (3 * x^2 - 1) / 2, x -> (5 * x^3 - 3 * x) / 2]

@test NonlinearBandits.legendre_next(0.5, 2, 0.5, 1.0) == polys[3](0.5)
@test_throws(DomainError("x lies outside [- 1 - 1e-4, 1 + 1e-4]"),
             NonlinearBandits.legendre_next(1.1, 0, 0.0, 0.0))
@test_throws(ArgumentError("tol must be non-negative"),
             NonlinearBandits.legendre_next(0.0, 2, 0.0, 0.0; tol=-1.0))
@test_throws(ArgumentError("j must be >= 2 for a 2nd order recurrance relation"),
             NonlinearBandits.legendre_next(0.0, 1, 0.0, 0.0))
