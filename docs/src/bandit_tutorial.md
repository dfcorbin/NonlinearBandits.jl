# Bandits Tutorial

# Constructing a driver

In order to run a contextual multi-armed bandit simulation, one must first construct an
[`AbstractDriver`](@ref), which manages how the policy interacts with the environment. The
simplest of the drivers is the [`StandardDriver`](@ref), which simply passes a batch of
contexts to the policy, recieves an array of actions and observes a batch of rewards.

To construct the driver, we must first define how the contexts and rewards are generated.
This can be done using from the class
[`AbstractContextSampler`](@ref)/[`AbstractRewardSampler`](@ref).

```@example tutorial
using NonlinearBandits

d = 2 # Number of features
limits = repeat([-1.0 1.0], d, 1)
mf = (x -> -10.0, x -> 10.0) # Expected reward

csampler = UniformContexts(limits)
rsampler = GaussianRewards(mf)

X = csampler(1)
println("Contexts: ", X)
r = rsampler(X, [1]) # Choose action 1
println("Rewards: ", r)
```

In the above code we have first defined a 2-dimensional context space, the lower/upper bounds 
of each feature are given by the columns of `limits`. The context sampler of type
[`UniformContexts`](@ref) samples contexts uniformly across the space. The reward sampler
of type [`GaussianRewards`](@ref) ouputs Gaussian rewards centered on the corresponding function
within `mf`. Notice that, in this case, we have chosen reward functions that are independent of
the contexts, but this need not, and often isn't, the case. We sample one context vector and
pass it to the reward sampler, where we have manually chosen the first action. This outputs
the observed reward.

To complete the driver, we need to supply an [`AbstractPolicy`](@ref) and an optional 
tuple, with elements of type [`AbstractMetric`]. For simplicity, we will construct a policy that
chooses action at random. We can also compute the regret of each decision using the
[`FunctionalRegret`](@ref) metric.

```@example tutorial
policy = RandomPolicy(d)
metrics = (FunctionalRegret(mf),)
driver = StandardDriver(csampler, policy, rsampler, metrics)
typeof(driver)
```

We can now output batches of observations from the driver.

```@example tutorial
batch_size = 5
X, a, r = driver(batch_size)
println("contexts: ", X)
println("actions: ", a)
println("rewards: ", r)

regret = metrics[1].regret
println("regret: ", regret)
```

Notice that regret is only non-zero when the second action is chosen. With the driver
constructed, we can finally run a batched simulation, where the policy is update (via
[`update!`](@ref)) after each batch.

```@example tutorial
using Plots
theme(:ggplot2)

num_batches = 10
run!(num_batches, batch_size, driver)

regret = metrics[1].regret
plot(regret, legend=nothing, ylab="regret", xlab="timestep")
```

There have been a total of 55 timesteps, since the driver has been called 11 times with a
batch size of 5 (including the call in the previous code block). The regret is purely
random over the course of the trajectory, since we always choose actions randomly.

# Example: PolynomialThompsonSampling

To see demonstrate the learning process of a bandit agent, we use the
[`PolynomialThompsonSampling`](@ref) policy. First construct the driver:

```@example poly
using NonlinearBandits, Plots, Colors
theme(:ggplot2)

d = 1
limits = repeat([-1.0 1.0], d, 1)
mf = (
    x -> 10 * sin(5^(2 * x[1]) * x[1]),
    x -> x[1]^3 - 3 * x[1]^2 + 4 * x[1]
)


xplt = -1.0:0.01:1.0
cls = distinguishable_colors(length(mf), RGB(0, 128/256, 128/256))
mf_plot = plot(legend=:topleft)
for i in 1:length(mf)
    plot!(xplt, x -> mf[i]([x]), label="Arm $i", color=cls[i], xlab="x", ylab="Expected Reward")
end
plot(mf_plot) # Display the reward functions
```

This is a difficult problem as the reward function for the teal arm is far smoother on the
left side of the space than the right. This can confuse the agent into believing the function
is smoother than it is.


```@example poly
num_arms = length(mf)
csampler = UniformContexts(limits)
rsampler = GaussianRewards(mf)

batch_size = 10 # Update linear model after ever 10 steps
inital_batches = 1 # Initialise models after 1 batch
retrain_freq = 10 # Retrain partition after every 10 batches
policy = PolynomialThompsonSampling(
    limits, 
    num_arms, 
    inital_batches, 
    retrain_freq;
    λ=15.0, # Increase prior scaling for complex functions
    α=15.0, # Increase exploration for difficult problem
    tol=1e-3, # Regulate complexity of partition
)
metrics = (FunctionalRegret(mf),)
driver = StandardDriver(csampler, policy, rsampler, metrics)

# The details of the below function are not relevant. This simply computes the standard
# deviation of the Thompson samples. This is used to visualize the "confidence" the agent
# has in an action's optimality.
function thompson_std(pol::PolynomialThompsonSampling, X::AbstractMatrix{Float64}, a::Int64)
    n = size(X, 2)
    σ = zeros(n)

    for i in 1:n
        shape, scale = NonlinearBandits.shape_scale(pol.arms[a])
        x = X[:, i:i]
        k = locate(pol.arms[a].P, x)[1]
        varmean = scale / (shape - 1)
        Σ = pol.α * varmean * pol.arms[a].models[k].lm.Σ
        z = expand(
            x,
            pol.arms[a].models[k].basis,
            pol.arms[a].P.regions[k];
            J=pol.arms[a].models[k].J,
        )
        σ[i] = (z' * Σ * z)[1, 1] |> sqrt
    end
    return σ
end

view_batches = [1, 10, 20, 50, 1000]
Xplt = reshape(xplt, (1, :))

plt_vec = []
for s in view_batches
    run!(s - policy.batches, batch_size, driver, verbose=false)
    plt = plot(legend=nothing, title="Batches: $s, Time: $(policy.t)")
    for a in 1:length(mf)
        plot!(plt, xplt, mf[a], color=cls[a])
        yplt = policy.arms[a](Xplt)
        std = thompson_std(policy, Xplt, a)
        plot!(xplt, yplt[1, :], color=cls[a], ribbon= 2 * std, ls=:dash, fillalpha=0.2)
        Xa, ra = arm_data(policy.data, a)
        plot!(Xa, ra, alpha=0.2, st=:scatter, color=cls[a])
    end
    push!(plt_vec, plt)
end
plot(plt_vec..., size=(700, length(view_batches) * 600), layout=(length(view_batches), 1))
```

From these plots we can see how the agent trials different actions across the space, then
eventually learns to choose the optimal actions depending on the context. This is evident
from the total regret:

```@example poly
total_regret = metrics[1].regret |> cumsum
plot(total_regret, ylab="Total regret", xlab="t", legend=nothing)
```