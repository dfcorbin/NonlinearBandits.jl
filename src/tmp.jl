using NonlinearBandits
f(x) = sin(10 * x[1]) + cos(10 * x[2])
X, y = NonlinearBandits.gaussian_data(2, 100, f)
m = PartitionedBayesPM(X, y, [0.0 1.0; 0.0 1.0]);