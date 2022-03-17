d = 2
data = BanditDataset(d)
X = rand(2, 2)
a = [1, 2]
r = rand(1, 2)
add_data!(data, X, a, r)
@test data.X == X
@test data.a == a
@test data.r == r
@test arm_data(data, 1) == (X[:, 1:1], r[:, 1:1])
@test arm_data(data, 2) == (X[:, 2:2], r[:, 2:2])