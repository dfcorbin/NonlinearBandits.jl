# Contextual Multi Armed Bandits

This tutorial will outline the extensible Contextual Multi Armed Bandit (CMAB) framework
implemented in [NonlinearBandits.jl](https://github.com/dfcorbin/NonlinearBandits.jl).

## Context Samplers

A context sampler is any callable object that takes an integer `n > 0` and outputs
a `d x n` matrix of contexts. For example, the simplest context sampler is one that
produces contexts uniformly across a predefined space.

```@example bandits
using NonlinearBandits

d = 5 # number of features in the context vectors
limits = repeat([-1.0 1.0], d, 1) # Defines a hyperrectangualr region
csampler = UniformContexts(limits)
csampler(2)
```

## Policies

A policy is any callable object that takes a `d x n` matrix as input, and outputs a
1-dimensional vector or integers indexing the actions. The simples example of a policy
is one that chooses actions at random.

```@example bandits
X = csampler(2) # Generate contexts
num_actions = 3
policy = RandomPolicy(num_actions)
policy(X)
```

# Reward Samplers

A reward sampler takes a `d x n` matrix of contexts, `X`, and a vector of actions, `a`,
and outputs the reward observed for each action. Gaussian distributed rewards can
be obtained by the following...

```@example bandits 
mf = (x -> -100, x -> 0,  x -> 100) # rewards independent of contexts in this case
rsampler = GaussianRewards(mf)

X = csampler(2) # Generate contexts
a = policy(X) # Generate actions
r = rsampler(X, a)
println("actions: ", a)
println("rewards: ", r)
```