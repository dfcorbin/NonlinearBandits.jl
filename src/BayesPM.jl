function legendre_next(x::AbstractFloat, j::Int, p1::AbstractFloat, p0::AbstractFloat;
                       tol::AbstractFloat=1e-4)
    if x < -1.0 - tol || x > 1.0 + tol
        throw(DomainError("x lies outside [- 1 - 1e-4, 1 + 1e-4]"))
    elseif tol < 0.0
        throw(ArgumentError("tol must be non-negative"))
    elseif j < 2
        throw(ArgumentError("j must be >= 2 for a 2nd order recurrance relation"))
    end
    x = min(x, 1.0)
    x = max(x, -1.0)
    return (2 * j - 1) * x * p1 / j - (j - 1) * p0 / j
end
