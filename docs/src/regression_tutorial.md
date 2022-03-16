# Model API

This tutorial outlines some of the available models in
[NonlinearBandits.jl](https://github.com/dfcorbin/NonlinearBandits.jl). First we generate some
synthetic data to work with.

```@example regression
using NonlinearBandits, Plots

d, n = 1, 500
f(x) = 10 * sin(5^(2 * x[1]) * x[1])
X, y = NonlinearBandits.gaussian_data(d, n, f)
data_plt = plot(X[1, :], y[1, :], label=nothing, alpha=0.3, legend=:topleft, st=:scatter)
```

## Bayesian Linear Model

A simple [Bayesian linear model](https://en.wikipedia.org/wiki/Bayesian_linear_regression) is
implemented by the [`BayesLM`](@ref) type.

```@example regression
lm = BayesLM(d)
fit!(lm, X, y)

xplt = 0:0.01:1
xplt = reshape(xplt, (d, :))
yplt = lm(xplt) # Call model object to make predictions
plot(data_plt, xplt[1, :], yplt[1, :], label="Linear model")
```

# Bayesian Polynomial Model

Using a [`BayesPM`](@ref) object, we can create a wrapper for the [`BayesLM`](@ref) class,
which first applies a polynomial expansion to the features.

```@example regression
limits = repeat([0.0 1.0], d, 1) # Define hyperrectangular limits of the feature space
basis = tpbasis(d, 3) # Use the degree-3 truncated polynomial tensor-product basis
pm = BayesPM(basis, limits)
fit!(pm, X, y)
yplt = pm(xplt) 
plot(data_plt, xplt[1, :], yplt[1, :], label="Polynoimal model")
```

# Partitioned Bayesian Polynomial Model

A more complex polynomial can be constructed by assigning low-degree polynomials
to disjoint regions of a partition. Suppose we wanted to partition our 1-dimensional
feature space into two disjoint regions, the partitioning api works as follows.

```@example regression
P = Partition(limits)
split!(P, 1, 1) # Split subregion 1 in dimension 1.
println("Region 1: ", P.regions[1])
println("Region 2: ", P.regions[2])
```

To construct the partitioned polynomial, given P:

```@example regression
ppm = PartitionedBayesPM(P, [3, 2])
fit!(ppm, X, y)
yplt = ppm(xplt)
plot(data_plt, xplt[1, :], yplt[1, :], label="Partitioned model")
```

Often a good choice of partition is not easy to construct by hand. In this case,
[`auto_partitioned_bayes_pm`](@ref) will perform a 1-step look ahead greedy search
to build the partition automatically. The degrees for each polnomial are chosen
automatically.

```@example regression
auto_ppm = auto_partitioned_bayes_pm(X, y, limits)
yplt = auto_ppm(xplt)
plot(data_plt, xplt[1, :], yplt[1, :], label="Partitioned model")
```