# Tutorial

This tutorial outlines some of the available models in
[NonlinearBandits.jl](https://github.com/dfcorbin/NonlinearBandits.jl). First we generate some
synthetic data to work with.

```@example regression
using NonlinearBandits, Plots

d, n = 1, 500
f(x) = 10 * sin(10^x[1] * x[1])
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
xplt = reshape(xplt, (1, :))
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