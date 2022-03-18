using CUDA, Flux, NonlinearBandits, Test

d = 5
num_actions = 2
layer_sizes = [64, 64]
encoder = NeuralEncoder(d, num_actions, layer_sizes)
@test size(encoder(rand(5, 2))) == (layer_sizes[end], 2)