# Test partitioning
d = 1
limits = repeat([0.0 1.0], d, 1)
P = Partition(limits)
split!(P, 1, 1)

@test P.regions[1] == [0.0 0.5]
@test P.regions[2] == [0.5 1.0]
@test locate(P, [0.0 0.5 1.0]) == [1, 2, 2]
@test_throws(ArgumentError("no region located for observation 2"), locate(P, [0.0 1.1]))

# Test variable selection 
d, n = 10, 1000
g(x) = 4 * x[5] - 10 * x[10] + x[2]
X, y = NonlinearBandits.gaussian_data(d, n, g)
@test lasso_selection(X, y, 2, false) == [5, 10]
@test lasso_selection(X, y, 2, true) == [1, 10]

# Set up a truly partitioned polynomial target function
d = 3
Jvals = [2, 3] # Degrees for left and right regions
basis = [tpbasis(d, J) for J in Jvals]
β = [rand(Normal(0, 10), length(basis[i]), 1) for i in 1:length(Jvals)]

limits = repeat([0.0 1.0], d, 1)
P = Partition(limits)
split!(P, 1, 1)

function f(x)
    x = reshape(x, (:, 1))
    k = locate(P, x)[1]
    z = expand(x, basis[k], P.regions[k]; J=Jvals[k])[[1, 3], :]
    return (z' * β[k][[1, 3], :])[1, 1]
end

n = 10000
X, y = NonlinearBandits.gaussian_data(d, n, f; σ=0.5)
pbpm = auto_partitioned_bayes_pm(X, y, limits; Pmax=2, verbose=false);

@test pbpm.P.regions == P.regions
@test mae(pbpm.models[1].lm.β, β[1][[1, 3]]) < 0.1
@test mae(pbpm.models[2].lm.β, β[2][[1, 3]]) < 0.1

X, y = NonlinearBandits.gaussian_data(d, 1, f; σ=0.5)
pbpm = auto_partitioned_bayes_pm(X, y, limits; verbose=false);
@test length(pbpm.models) == 1
@test pbpm.models[1].J == 0
