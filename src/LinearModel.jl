struct HyperParam
    shape::Float64
    scale::Float64
    prec::Symmetric{Float64}
    coefs::Vector{Float64}
end


mutable struct LinearModel
    prior::HyperParam
    post::HyperParam
end


function LinearModel(
    d::Int64;
    prior_shape::Float64 = 0.01,
    prior_scale::Float64 = 0.01,
    regularization::Float64 = 1.0,
)
    coefs = zeros(d)
    prec = Symmetric(diagm(fill(regularization, d)))
    prior = HyperParam(prior_shape, prior_scale, prec, coefs)
    post = deepcopy(prior)
    return LinearModel(prior, post)
end


function get_coefs(lm::LinearModel)
    return lm.post.coefs
end


function get_shape(lm::LinearModel)
    return lm.post.shape
end


function get_scale(lm::LinearModel)
    return lm.post.scale
end


function get_prec(lm::LinearModel)
    return lm.post.prec
end


function get_prior_prec(lm::LinearModel)
    return lm.prior.prec
end


function variance(lm::LinearModel)
    if lm.post.shape <= 1
        msg = "mean of inverse gamma distribution is undefined for shape ≤ 1"
        DomainError(lm.post.shape, msg)
    end
    return lm.post.scale / (lm.post.shape - 1)
end


function fit!(lm::LinearModel, X::Matrix{Float64}, y::Vector{Float64})
    prec = Symmetric(X' * X) + lm.post.prec
    target = X' * y + lm.post.prec * lm.post.coefs
    chol = cholesky(prec)
    coefs = chol.U \ (chol.L \ target)
    shape = lm.post.shape + 0.5 * length(y)
    scale = lm.post.scale + 0.5 * y' * y
    scale += 0.5 * lm.post.coefs' * lm.post.prec * lm.post.coefs
    scale -= 0.5 * coefs' * prec * coefs
    lm.post = HyperParam(shape, scale, prec, coefs)
    return nothing
end


function (lm::LinearModel)(X::Matrix{Float64})
    return X * lm.post.coefs
end


function posterior_sample(lm::LinearModel, x::AbstractVector, inflation::Float64 = 1.0)
    s2 = rand(InverseGamma(lm.post.shape, lm.post.scale))
    prec = lm.post.prec / (s2 * inflation)
    coefs = rand(MvNormalCanon(prec * lm.post.coefs, prec))
    return coefs' * x
end