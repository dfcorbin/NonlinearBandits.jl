# Test partitioning
d = 1
limits = repeat([0.0 1.0], d, 1)
P = Partition(limits)
split!(P, 1, 1)
@test_throws(ArgumentError("P does not contain a subregion at index 3"), split!(P, 3, 1))
@test_throws(ArgumentError("P is not 2-dimensional"), split!(P, 1, 2))
@test P.regions[1] == [0.0 0.5]
@test P.regions[2] == [0.5 1.0]
@test locate(P, [0.0 0.5 1.0]) == [1, 2, 2]
@test_throws(ArgumentError("no region located for observation 2"), locate(P, [0.0 1.1]))
@test_throws(
    DimensionMismatch("P does match the dimension of X"), locate(P, [0.0 0.0; 0.0 0.0])
)

# Test variable selection 
d, n = 10, 1000
g(x) = 4 * x[5] - 10 * x[10] + x[2]
X, y = NonlinearBandits.gaussian_data(d, n, g)
@test lasso_selection(X, y, 2, false) == [5, 10]
@test lasso_selection(X, y, 2, true) == [1, 10]

# Set up a truly partitioned polynomial target function
d, n = 3, 10000
Jvals = [2, 3] # Degrees for left and right regions
basis = [tpbasis(d, J) for J in Jvals]
β = [rand(Normal(0, 10), length(basis[i]), 1) for i in 1:length(Jvals)]
limits = repeat([0.0 1.0], d, 1)
P = Partition(limits)
split!(P, 1, 1)

function f(x)
    # Sparse basis is used to test lasso selection
    x = reshape(x, (:, 1))
    k = locate(P, x)[1]
    z = expand(x, basis[k], P.regions[k]; J=Jvals[k])
    return (z' * β[k])[1, 1]
end

X, y = NonlinearBandits.gaussian_data(d, n, f; σ=0.5)
pbpm = PartitionedBayesPM(X, y, limits; verbose=false);
@test pbpm.P.regions == P.regions
@test mae(pbpm.models[1].lm.β, β[1]) < 0.1
@test mae(pbpm.models[2].lm.β, β[2]) < 0.1

@test_throws(
    ArgumentError("X and limits don't match in their first dimension"),
    PartitionedBayesPM(X[1:2, :], y, limits; verbose=false)
)
@test_throws(
    ArgumentError("Jmax must be non-negative"),
    PartitionedBayesPM(X, y, limits; Jmax=-1, verbose=false)
)
@test_throws(
    ArgumentError("Kmax must be strictly positive"),
    PartitionedBayesPM(X, y, limits; Kmax=0, verbose=false)
)
@test_throws(
    ArgumentError("tolerance must be non-negative"),
    PartitionedBayesPM(X, y, limits; tol=-1.0)
)

# Test manual partitioned model setup
function g(x)
    x = reshape(x, (:, 1))
    k = locate(P, x)[1]
    z = expand(x, basis[k], P.regions[k]; J=Jvals[k])
    return (z' * β[k])[1, 1]
end

X, y = NonlinearBandits.gaussian_data(d, n, g; σ=0.5)
manual_pbpm = PartitionedBayesPM(P, Jvals)
fit!(manual_pbpm, X, y)
@test mae(manual_pbpm.models[1].lm.β, β[1]) < 0.1
@test mae(manual_pbpm.models[2].lm.β, β[2]) < 0.1
@test_throws(
    ArgumentError("must supply a value of J for every region in P"),
    PartitionedBayesPM(P, [1])
)

@test mae(manual_pbpm(X), mapslices(g, X; dims=1)) < 0.1

sequential_ppm = PartitionedBayesPM(P, Jvals)
for i in 1:size(X, 2)
    fit!(sequential_ppm, X[:, i:i], y[:, i:i])
end
@test sequential_ppm.shape == 0.001 + size(X, 2) / 2
@test sequential_ppm.scale ≈ manual_pbpm.scale