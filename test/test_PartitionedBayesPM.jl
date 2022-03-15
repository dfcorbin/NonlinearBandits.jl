using NonlinearBandits
using Test
using Distributions: Uniform, Normal
import Base: ==

# Test partitioning
d = 1
limits = repeat([0.0 1.0], d, 1)
P = Partition(limits)
split!(P, 1, 1)

@test P.regions[1] == [0.0 0.5]
@test P.regions[2] == [0.5 1.0]
@test locate(P, [0.0 0.5 1.0]) == [1, 2, 2]
@test_throws(ArgumentError("no region located for observation 2"), locate(P, [0.0 1.1]))

# Set up a truly partitioned polynomial target function
Jvals = [0, 2] # Degrees for left and right regions
basis = [tpbasis(d, J) for J in Jvals]
β = [rand(Normal(0, 10), length(basis[i]), 1) for i in 1:length(Jvals)]

function f(x)
    x = reshape(x, (:, 1))
    k = locate(P, x)[1]
    z = expand(x, basis[k], P.regions[k]; J=Jvals[k])
    return (z' * β[k])[1, 1]
end

n = 10000
X, y = NonlinearBandits.gaussian_data(d, n, f; σ=0.5)

pbpm = auto_partitioned_bayes_pm(X, y, limits);
