function gaussian_data(d, n, f; σ=1)
    X = rand(d, n)
    y = mapslices(f, X; dims=1) + rand(Normal(0, σ), 1, n)
    return X, y
end
