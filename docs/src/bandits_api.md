# Bandit API

```@docs
update!(pol::AbstractPolicy, X::AbstractMatrix{Float64}, a::AbstractVector{Int64},
        r::AbstractMatrix{Float64})
```

# Context Samplers

```@autodocs
Modules = [NonlinearBandits]
Pages = ["contexts.jl"]
```

# Policies

```@autodocs
Modules = [NonlinearBandits]
Pages = ["RandomPolicy.jl", "PolynomialThompsonSampling.jl"]
```

# Reward Samplers

```@autodocs
Modules = [NonlinearBandits]
Pages = ["rewards.jl"]
```

# Metrics

```@autodocs
Modules = [NonlinearBandits]
Pages = ["metrics.jl"]
```

# Drivers

```@autodocs
Modules = [NonlinearBandits]
Pages = ["drivers.jl"]
```