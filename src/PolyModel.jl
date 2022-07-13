struct Polynomial
    degrees::Vector{Int64}
    coefs::Vector{Float64}
end


function legendre(degree::Int64)
    num_coefs = Int(floor(degree / 2))
    degrees = [degree - 2 * i for i in 0:num_coefs]
    coefs = [
        2.0^(-degree) *
        (-1.0)^i * binomial(degree, i) *
        binomial(2 * degree - 2 * i, degree)
        for i in 0:num_coefs
    ]
    return Polynomial(degrees, coefs)
end