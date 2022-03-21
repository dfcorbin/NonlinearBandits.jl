var documenterSearchIndex = {"docs":
[{"location":"bandit_tutorial/#Bandits-Tutorial","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"","category":"section"},{"location":"bandit_tutorial/#Constructing-a-driver","page":"Bandits Tutorial","title":"Constructing a driver","text":"","category":"section"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"In order to run a contextual multi-armed bandit simulation, one must first construct an AbstractDriver, which manages how the policy interacts with the environment. The simplest of the drivers is the StandardDriver, which simply passes a batch of contexts to the policy, recieves an array of actions and observes a batch of rewards.","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"To construct the driver, we must first define how the contexts and rewards are generated. This can be done using objects with parent classes AbstractContextSampler/AbstractRewardSampler.","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"using NonlinearBandits\n\nd = 2 # Number of features\nlimits = repeat([-1.0 1.0], d, 1)\nmf = (x -> -10.0, x -> 10.0) # Expected reward\n\ncsampler = UniformContexts(limits)\nrsampler = GaussianRewards(mf)\n\nX = csampler(1)\nprintln(\"Contexts: \", X)\nr = rsampler(X, [1]) # Choose action 1\nprintln(\"Rewards: \", r)","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"In the above code we have first defined a 2-dimensional context space, the lower/upper bounds  of each feature are given by the columns of limits. The context sampler of type UniformContexts samples contexts uniformly across the space. The reward sampler of type GaussianRewards ouputs Gaussian rewards centered on the corresponding function within mf. Notice that, in this case, we have chosen reward functions that are independent of the contexts, but this need not, and often isn't, the case. We sample one context vector and pass it to the reward sampler, where we have manually chosen the first action. This outputs the observed reward.","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"To complete the driver, we need to supply an AbstractPolicy and an optional  tuple, with elements of type AbstractMetric. For simplicity, we will construct a policy that chooses action at random. We can also compute the regret of each decision using the FunctionalRegret metric.","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"policy = RandomPolicy(d)\nmetrics = (FunctionalRegret(mf),)\ndriver = StandardDriver(csampler, policy, rsampler, metrics)\ntypeof(driver)","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"We can now output batches of observations from the driver.","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"batch_size = 5\nX, a, r = driver(batch_size)\nprintln(\"contexts: \", X)\nprintln(\"actions: \", a)\nprintln(\"rewards: \", r)\n\nregret = metrics[1].regret\nprintln(\"regret: \", regret)","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"Notice that regret is only non-zero when the second action is chosen. With the driver constructed, we can finally run a batched simulation, where the policy is update (via update!) after each batch.","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"using Plots\ntheme(:ggplot2)\n\nnum_batches = 10\nrun!(num_batches, batch_size, driver)\n\nregret = metrics[1].regret\nplot(regret, legend=nothing, ylab=\"regret\", xlab=\"timestep\")","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"There have been a total of 55 timesteps, since the driver has been called 11 times with a batch size of 5 (including the call in the previous code block). The regret is purely random over the course of the trajectory, since we always choose actions randomly.","category":"page"},{"location":"bandit_tutorial/#Example:-PolynomialThompsonSampling","page":"Bandits Tutorial","title":"Example: PolynomialThompsonSampling","text":"","category":"section"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"To see demonstrate the learning process of a bandit agent, we use the PolynomialThompsonSampling policy. First construct the driver:","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"using NonlinearBandits, Plots, Colors\ntheme(:ggplot2)\n\nd = 1\nlimits = repeat([-1.0 1.0], d, 1)\nmf = (\n    x -> 10 * sin(5^(2 * x[1]) * x[1]),\n    x -> x[1]^3 - 3 * x[1]^2 + 4 * x[1]\n)\n\n\nxplt = -1.0:0.01:1.0\ncls = distinguishable_colors(length(mf), RGB(0, 128/256, 128/256))\nmf_plot = plot(legend=:topleft)\nfor i in 1:length(mf)\n    plot!(xplt, x -> mf[i]([x]), label=\"Arm $i\", color=cls[i], xlab=\"x\", ylab=\"Expected Reward\")\nend\nplot(mf_plot) # Display the reward functions","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"This is a difficult problem as the reward function for the teal arm is far smoother on the left side of the space than the right. This can confuse the agent into believing the function is smoother than it is.","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"num_arms = length(mf)\ncsampler = UniformContexts(limits)\nrsampler = GaussianRewards(mf)\n\nbatch_size = 10 # Update linear model after ever 10 steps\ninital_batches = 1 # Initialise models after 1 batch\nretrain_freq = 10 # Retrain partition after every 10 batches\npolicy = PolynomialThompsonSampling(\n    limits, \n    num_arms, \n    inital_batches, \n    retrain_freq;\n    λ=20.0, # Increase prior scaling for complex functions\n    α=20.0, # Increase exploration for difficult problem\n    tol=1e-3, # Regulate complexity of partition\n)\nmetrics = (FunctionalRegret(mf),)\ndriver = StandardDriver(csampler, policy, rsampler, metrics)\n\n# The details of the below function are not relevant. This simply computes the standard\n# deviation of the Thompson samples. This is used to visualize the \"confidence\" the agent\n# has in an action's optimality.\nfunction thompson_std(pol::PolynomialThompsonSampling, X::AbstractMatrix{Float64}, a::Int64)\n    n = size(X, 2)\n    σ = zeros(n)\n\n    for i in 1:n\n        shape, scale = NonlinearBandits.shape_scale(pol.arms[a])\n        x = X[:, i:i]\n        k = locate(pol.arms[a].P, x)[1]\n        varmean = scale / (shape - 1)\n        Σ = pol.α * varmean * pol.arms[a].models[k].lm.Σ\n        z = expand(\n            x,\n            pol.arms[a].models[k].basis,\n            pol.arms[a].P.regions[k];\n            J=pol.arms[a].models[k].J,\n        )\n        σ[i] = (z' * Σ * z)[1, 1] |> sqrt\n    end\n    return σ\nend\n\nview_batches = [1, 10, 20, 50, 1000]\nXplt = reshape(xplt, (1, :))\n\nplt_vec = []\nfor s in view_batches\n    run!(s - policy.batches, batch_size, driver, verbose=false)\n    plt = plot(legend=nothing, title=\"Batches: $s, Time: $(policy.t)\")\n    for a in 1:length(mf)\n        plot!(plt, xplt, mf[a], color=cls[a])\n        yplt = policy.arms[a](Xplt)\n        std = thompson_std(policy, Xplt, a)\n        plot!(xplt, yplt[1, :], color=cls[a], ribbon= 2 * std, ls=:dash, fillalpha=0.2)\n        Xa, ra = arm_data(policy.data, a)\n        plot!(Xa, ra, alpha=0.2, st=:scatter, color=cls[a])\n    end\n    push!(plt_vec, plt)\nend\nplot(plt_vec..., size=(700, length(view_batches) * 600), layout=(length(view_batches), 1))","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"From these plots we can see how the agent trials different actions across the space, then eventually learns to choose the optimal actions depending on the context. This is evident from the total regret:","category":"page"},{"location":"bandit_tutorial/","page":"Bandits Tutorial","title":"Bandits Tutorial","text":"total_regret = metrics[1].regret |> cumsum\nplot(total_regret, ylab=\"Total regret\", xlab=\"t\", legend=nothing)","category":"page"},{"location":"model_api/#Model-API","page":"Models","title":"Model API","text":"","category":"section"},{"location":"model_api/","page":"Models","title":"Models","text":"fit!(model, X::AbstractMatrix, y::AbstractMatrix)","category":"page"},{"location":"model_api/#NonlinearBandits.fit!-Tuple{Any, AbstractMatrix, AbstractMatrix}","page":"Models","title":"NonlinearBandits.fit!","text":"fit!(model, X::AbstractMatrix, y::AbstractMatrix)\n\nUpdate the parameters of model.\n\nArguments\n\nX::AbstractMatrix: A matrix with observations stored as columns.\ny::AbstractMatrix: A matrix with 1 row of response variables. \n\n\n\n\n\n","category":"method"},{"location":"model_api/#Polynomials","page":"Models","title":"Polynomials","text":"","category":"section"},{"location":"model_api/","page":"Models","title":"Models","text":"Modules = [NonlinearBandits]\nPages = [\"BayesLM.jl\", \"BayesPM.jl\", \"PartitionedBayesPM.jl\"]","category":"page"},{"location":"model_api/#NonlinearBandits.AbstractBayesianLM","page":"Models","title":"NonlinearBandits.AbstractBayesianLM","text":"Abstract type for models using a Gaussian/normal-inverse-gamma conjugate prior.\n\n\n\n\n\n","category":"type"},{"location":"model_api/#NonlinearBandits.BayesLM","page":"Models","title":"NonlinearBandits.BayesLM","text":"BayesLM(d::Int; <keyword arguments>)\n\nConstruct a Bayesian linear model.\n\nArguments\n\nλ::Float64=1.0: Prior scaling.\nshape0::Float64=1e-3: Inverse-gamma prior shape hyperparameter.\nscale0::Float64=1e-3: Inverse-gamma prior scale hyperparameter.\n\n\n\n\n\n","category":"type"},{"location":"model_api/#NonlinearBandits.std-Tuple{AbstractBayesianLM}","page":"Models","title":"NonlinearBandits.std","text":"std(model::AbstractBayesianLM)\n\nReturn the posterior mean of the inverse-gamma distribution.\n\n\n\n\n\n","category":"method"},{"location":"model_api/#NonlinearBandits.BayesPM","page":"Models","title":"NonlinearBandits.BayesPM","text":"BayesPM(basis::Vector{Index}, limits::Matrix{Float64}; <keyword arguments>)\n\nConstruct a Bayesian linear model on polynomial features.\n\nArguments\n\nbasis::Vector{Index}: Vector of monomial indices.\nlimits::Matrix{Float64}: Matrix with two columns defining the lower/upper limits of the space.\nλ::Float64=1.0: Prior covariance scale factor.\nshape0::Float64=1e-3: Inverse-gamma prior shape hyperparameter.\nscale0::Float64=1e=3: Inverse-gamma prior scale hyperparameter.\n\n\n\n\n\n","category":"type"},{"location":"model_api/#NonlinearBandits.Index","page":"Models","title":"NonlinearBandits.Index","text":"Index(dim::Vector{Int64}, deg::Vector{Int64})\n\nMultivariate monomial index.\n\nThe monomial x[1] * x[3]^2can be encoded usingdim = [1, 3],deg = [1, 2]`\n\n\n\n\n\n","category":"type"},{"location":"model_api/#NonlinearBandits.Partition","page":"Models","title":"NonlinearBandits.Partition","text":"Partition(limits::Matrix{Float64})\n\nConstruct an object capable of storing a hyperrectangular partition.\n\n\n\n\n\n","category":"type"},{"location":"model_api/#NonlinearBandits.PartitionedBayesPM-Tuple{AbstractMatrix, AbstractMatrix, Matrix{Float64}}","page":"Models","title":"NonlinearBandits.PartitionedBayesPM","text":"PartitionedBayesPM(X::AbstractMatrix, y::AbstractMatrix, limits::Matrix{Float64};\n                   <keyword arguments>)\n\nPerform a 1-step look ahead greedy search for a partitioned polynomial model.\n\nKeyword Arguments\n\nJmax::Int64=3: The maximum degree of any polynomial model.\nPmax::Int64=500: The maximum number of features in a particular regions.\nKmax::Int64=200: The maximum number of regions\nλ::Float64=1.0: Prior scaling.\nshape0::Float64=1e-3: Inverse-gamma prior shape hyperparameter.\nscale0::Float64=1e-3: Inverse-gamma prior scale hyperparameter.\nratio::Float64=1.0: Polynomial degrees are reduced until size(X, 2) < ratio * length(tpbasis(d, J)).\ntol::Float64=1e-4: The required increase in the model evidence to accept a split.\nverbose::Bool=true: Print details of the partition search.\n\n\n\n\n\n","category":"method"},{"location":"model_api/#NonlinearBandits.PartitionedBayesPM-Tuple{Partition, Vector{Int64}}","page":"Models","title":"NonlinearBandits.PartitionedBayesPM","text":"PartitionedBayesPM(P::Partition, Js::Vector{Int64}; <keyword arguments>)\n\nContruct a partitioned polynomial model.\n\nArguments\n\nP::Partition: A partition of the space.\nλ::Float64=1.0: Prior scaling.\nshape0::Float64=1e-3: Inverse-gamma prior shape hyperparameter.\nscale0::Float64=1e-3: Inverse-gamma prior scale hyperparameter.\n\n\n\n\n\n","category":"method"},{"location":"model_api/#NonlinearBandits.expand-Tuple{AbstractMatrix, Vector{Index}, Matrix{Float64}}","page":"Models","title":"NonlinearBandits.expand","text":"expand(X::AbstractMatrix, basis::Vector{Index}, limits::Matrix{Float64};\n       J::Union{Nothing,Int64}=nothing)\n\nExpand the columns of X into a rescaled legendre polynomial basis.\n\nArguments\n\nX::AbstractMatrix: Matrix with observations stored as columns.\nbasis::Vector{Index}: Vector of monomial indices.\nlimits::Matrix{Float64}: Matrix with two columns defining the lower/upper limits of the space.\nJ::Union{Nothing, Int64}=nothing: The maximum degree of the basis. Inferred if not specified.\n\n\n\n\n\n","category":"method"},{"location":"model_api/#NonlinearBandits.lasso_selection-Tuple{AbstractMatrix, AbstractMatrix, Int64, Bool}","page":"Models","title":"NonlinearBandits.lasso_selection","text":"lasso_selection(X::AbstractMatrix, y::AbstractMatrix, Pmax::Int64, intercept::Bool)\n\nChoose the first Pmax features introduced by a LASSO solution path.\n\nArguments\n\nX::AbstractMatrix: A matrix with observations stored as columns.\ny::AbstractMatrix: A matrix with 1 row of response variables. \nPmax::Int64: The maximum number of predictors.\nintercept::Bool: true if the first row of X are the intercept features\n\n\n\n\n\n","category":"method"},{"location":"model_api/#NonlinearBandits.locate-Tuple{Partition, AbstractMatrix}","page":"Models","title":"NonlinearBandits.locate","text":"locate(P::Partition, X::AbstractMatrix)\n\nReturn a vector of integers giving the region index for each column of X.\n\n\n\n\n\n","category":"method"},{"location":"model_api/#NonlinearBandits.split!-Tuple{Partition, Int64, Int64}","page":"Models","title":"NonlinearBandits.split!","text":"split!(P::Partition, k::Int64, d::Int64)\n\nSplit the k'th subregion of P into equal halves in dimension d.\n\n\n\n\n\n","category":"method"},{"location":"model_api/#NonlinearBandits.tpbasis-Tuple{Int64, Int64}","page":"Models","title":"NonlinearBandits.tpbasis","text":"tpbasis(d::Int64, J::Int64)\n\nConstruct the d-dimensional truncated tensor-product basis.\n\nAll index terms have a degree ≤ J.\n\nSee also Index\n\n\n\n\n\n","category":"method"},{"location":"model_api/","page":"Models","title":"Models","text":"shape_scale(model::AbstractBayesianLM)","category":"page"},{"location":"model_api/#NonlinearBandits.shape_scale-Tuple{AbstractBayesianLM}","page":"Models","title":"NonlinearBandits.shape_scale","text":"shape_scale(model::AbstractBayesianLM)\n\nReturn the shape/scale of model.\n\n\n\n\n\n","category":"method"},{"location":"#NonlinearBandits.jl","page":"Introduction","title":"NonlinearBandits.jl","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"NonlinearBandits.jl provides an easy to use framework for implementing contextual multi-armed bandit policies, and testing them with synthetic environments. Pre-implemented policies can be found in the Bandits API which may be useful to compare with.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"I recommend reading through the Tutorials (particularly the Bandits Tutorial) as a good starting point.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"","category":"page"},{"location":"bandits_api/#Bandits-API","page":"Bandits","title":"Bandits API","text":"","category":"section"},{"location":"bandits_api/","page":"Bandits","title":"Bandits","text":"Modules = [NonlinearBandits]\nPages = [\"bandits.jl\"]","category":"page"},{"location":"bandits_api/#NonlinearBandits.AbstractContextSampler","page":"Bandits","title":"NonlinearBandits.AbstractContextSampler","text":"Called with an integer n > 0 to genrate a d x n matrix of contexts.\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#NonlinearBandits.AbstractDriver","page":"Bandits","title":"NonlinearBandits.AbstractDriver","text":"Driver to manage how the bandit policy interacts with its environment.\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#NonlinearBandits.AbstractMetric","page":"Bandits","title":"NonlinearBandits.AbstractMetric","text":"A metric that will be called as metric(X, a, r) after each batch outputted by a driver.\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#NonlinearBandits.AbstractPolicy","page":"Bandits","title":"NonlinearBandits.AbstractPolicy","text":"A callable policy that outputs actions via policy(X) and is updated via update!\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#NonlinearBandits.AbstractRewardSampler","page":"Bandits","title":"NonlinearBandits.AbstractRewardSampler","text":"A callable object to ouput rewards via sampler(X, a).\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#NonlinearBandits.BanditDataset","page":"Bandits","title":"NonlinearBandits.BanditDataset","text":"BanditDataset(d::Int64)\n\nStores the trajectory of a bandit simulation.\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#NonlinearBandits.append_data!-Tuple{BanditDataset, AbstractMatrix, AbstractVector{<:Int64}, AbstractMatrix}","page":"Bandits","title":"NonlinearBandits.append_data!","text":"append_data!(data::BanditDataset, X::AbstractMatrix, a::AbstractVector{<:Int},\n          r::AbstractMatrix)\n\nAdd a batch of data to the dataset.\n\n\n\n\n\n","category":"method"},{"location":"bandits_api/#NonlinearBandits.arm_data-Tuple{BanditDataset, Int64}","page":"Bandits","title":"NonlinearBandits.arm_data","text":"rm_data(data::BanditDataset, a::Int64)\n\nReturn the data associated with arm a.\n\n\n\n\n\n","category":"method"},{"location":"bandits_api/#Context-Samplers","page":"Bandits","title":"Context Samplers","text":"","category":"section"},{"location":"bandits_api/","page":"Bandits","title":"Bandits","text":"Modules = [NonlinearBandits]\nPages = [\"contexts.jl\"]","category":"page"},{"location":"bandits_api/#NonlinearBandits.UniformContexts","page":"Bandits","title":"NonlinearBandits.UniformContexts","text":"UniformContexts(limits::Matrix{Float64})\n\nConstruct a callable object to generate uniform contexts.\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#Policies","page":"Bandits","title":"Policies","text":"","category":"section"},{"location":"bandits_api/","page":"Bandits","title":"Bandits","text":"update!","category":"page"},{"location":"bandits_api/#NonlinearBandits.update!","page":"Bandits","title":"NonlinearBandits.update!","text":"update!(pol::AbstractPolicy, X::AbstractMatrix, a::AbstractVector{<:Int}, \n        r::AbstractMatrix)\n\nUpdate pol with a batch of data.\n\n\n\n\n\n","category":"function"},{"location":"bandits_api/","page":"Bandits","title":"Bandits","text":"Modules = [NonlinearBandits]\nPages = [\"RandomPolicy.jl\", \"PolynomialThompsonSampling.jl\", \"NeuralLinear.jl\"]","category":"page"},{"location":"bandits_api/#NonlinearBandits.RandomPolicy","page":"Bandits","title":"NonlinearBandits.RandomPolicy","text":"RandomPolicy(num_actions::Int64)\n\nConstruct a policy that chooses actions at random.\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#NonlinearBandits.PolynomialThompsonSampling","page":"Bandits","title":"NonlinearBandits.PolynomialThompsonSampling","text":"PolynomialThompsonSampling(d::Int64, num_arms::Int64, initial_batches::Int64,\n                           retrain_freq::Vector{Int64}; <keyword arguments>)\n\nConstruct a Thompson sampling policy that uses a PartitionedBayesPM to model the expected rewards.\n\nArguments\n\nd::Int64: The number of features.\nnum_arms::Int64: The number of available actions.\ninital_batches::Int64: The number of batches to sample before training the polnomial   models.\nretrain::Vector{Int64}: The frequency (in terms of batches) at which the partition/basis   selection is retrained from scratch.\nα::Float64=1.0: Thompson sampling inflation. α > 1 and increasing alpha increases the   amount of exploration.\nJmax::Int64=3: The maximum degree of any polynomial region.\nPmax::Int64=100: The maximum number of features in any polynomial region.\nKmax::Int64=500: The maximum number of regions in the partition.\nλ::Float64=1.0: Prior scaling.\nshape0::Float64=1e-3: Inverse-gamma prior shape hyperparameter.\nscale0::Float64=1e-3: Inverse-gamma prior scale hyperparameter.\nratio::Float64=1.0: Polynomial degrees are reduced until size(X, 2) < ratio * length(tpbasis(d, J)).\ntol::Float64=1e-4: The required increase in the model evidence to accept a split.\nverbose_retrain::Bool=true: Print details of the partition search.\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#NonlinearBandits.NeuralEncoder","page":"Bandits","title":"NonlinearBandits.NeuralEncoder","text":"NeuralEncoder(d_int::Int64, d_out::Int64, layer_sizes::Vector{Int64})\n\nConstruct a callable object that returns the final layer activations of a neural network (which can be trained with a contextual multi armed bandit trajectory).\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#NonlinearBandits.NeuralLinear","page":"Bandits","title":"NonlinearBandits.NeuralLinear","text":"NeuralLinear(d::Int64, num_arms::Int64, layer_sizes::Vector{Int64}, inital_batches::Int64,\n             retrain_freq::Int64, epochs::Int64)\n\nNeuralLinear policy introduced in the paper Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling.\n\nKeyword Arguments\n\nopt=ADAM(): Optimizer used to update the neural network parameters.\nλ::Float64=1.0: Prior scaling.\nshape0::Float64=1e-3: Inverse-gamma prior shape hyperparameter.\nscale0::Float64=1e-3: Inverse-gamma prior scale hyperparameter.\nverbose_retrain::Bool=false: Print the details of the neural network training   procedure.\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#NonlinearBandits.fit!-Tuple{NeuralEncoder, AbstractMatrix, AbstractVector{<:Int64}, AbstractMatrix, Int64}","page":"Bandits","title":"NonlinearBandits.fit!","text":"fit!(enc::NeuralEncoder, X::AbstractMatrix, a::AbstractVector{<:Int}, r::AbstractMatrix,\n     epochs::Int64)\n\nUpdate a the neural network parameters using a contextual multi-armed bandit trajectory.\n\nKeyword Arguments\n\nbatch_size::Int64=32: The batch size to use while training the neural network.\nopt=ADAM(): The optimizer to use for updating the parameters.\nverbose::Bool=true: Print details of the fitting procedure.\n\n\n\n\n\n","category":"method"},{"location":"bandits_api/#Reward-Samplers","page":"Bandits","title":"Reward Samplers","text":"","category":"section"},{"location":"bandits_api/","page":"Bandits","title":"Bandits","text":"Modules = [NonlinearBandits]\nPages = [\"rewards.jl\"]","category":"page"},{"location":"bandits_api/#NonlinearBandits.GaussianRewards","page":"Bandits","title":"NonlinearBandits.GaussianRewards","text":"GaussianRewards(mf::Tuple{Vararg{<:Function}}; <keyword arguments>)\n\nConstruct a callable object to sample gaussian rewards.\n\nArguments\n\nmf::Tuple{Vararg{<:Function}}: A Tuple of functions which take a 1-dimensional input and   output the (scalar) mean reward for the corresponding action.\nσ::Float64: The standard deviation of the gaussian noise applied to each reward. \n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#Metrics","page":"Bandits","title":"Metrics","text":"","category":"section"},{"location":"bandits_api/","page":"Bandits","title":"Bandits","text":"Modules = [NonlinearBandits]\nPages = [\"metrics.jl\"]","category":"page"},{"location":"bandits_api/#NonlinearBandits.FunctionalRegret","page":"Bandits","title":"NonlinearBandits.FunctionalRegret","text":"FunctionalRegret(mf::Tuple{Vararg{<:Function}})\n\nMetric to track the regret of each action using a discrete set of functions.\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#Drivers","page":"Bandits","title":"Drivers","text":"","category":"section"},{"location":"bandits_api/","page":"Bandits","title":"Bandits","text":"Modules = [NonlinearBandits]\nPages = [\"drivers.jl\"]","category":"page"},{"location":"bandits_api/#NonlinearBandits.StandardDriver","page":"Bandits","title":"NonlinearBandits.StandardDriver","text":"StandardDriver(csampler::AbstractContextSampler, policy::AbstractPolicy, \n               rsampler::AbstractRewardSampler[, \n               metrics::Tuple{Vararg{<:AbstractMetric}}])\n\nA simple driver that samples contexts, passes them to the policy to generate actions, then observes the rewards.\n\nArguments\n\ncsampler::AbstractContextSampler: Context sampler.\npolicy::AbstractPolicy: Policy to generate actions given contexts.\nrsampler::AbstractRewardSampler: Sampler for rewards, given the contexts and actions.\nmetrics::Tuple{Vararg{<:AbstractMetric}}: A tuple of metrics that will each be    called as metric(X, a, r).\n\n\n\n\n\n","category":"type"},{"location":"bandits_api/#NonlinearBandits.run!-Tuple{Int64, Int64, AbstractDriver}","page":"Bandits","title":"NonlinearBandits.run!","text":"run!(num_batches::Int64, batch_size::Int64, driver::AbstractDriver; <keyword arguments>)\n\nRun a driver for num_batches batches.\n\n\n\n\n\n","category":"method"}]
}
