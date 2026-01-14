import torch
import numpy as np

# Batch dimension of tensors must match for matmul op.
# second to last dimension of tensorB must match with last dimension of tensorA
tensorA = torch.zeros(size=(6, 4, 5))
tensorB = torch.ones(size=(6, 5, 9))

# Rank 3 tensors must use matmul or @ in order to multiply
print(tensorA @ tensorB)
x = torch.arange(1.0, 8.0)
print(x, x.shape)

x_reshape = x.reshape(1, 7)
print(f"x.reshape: {x_reshape}")

rand_to_squeeze = torch.rand(size=(224, 224, 3))
print(rand_to_squeeze, rand_to_squeeze.size())
rand_squeezed = rand_to_squeeze.squeeze()
print(rand_squeezed, rand_squeezed.size())

x = torch.arange(1, 10)
print(x)
x = x.reshape(1, 3, 3)
print(x)

# Pytorch and numpy
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array)
print(tensor)

# Reproducibility and randomness in Pytorch
random_tensorA = torch.rand(3, 4)
random_tensorB = torch.rand(3, 4)

print(f"Tensor A:\n{random_tensorA}\n")
print(f"Tensor B:\n{random_tensorB}\n")
print("Does tensor A equal tensor B (anywhere)")
print(random_tensorA == random_tensorB)

rand_4d_tensor = torch.rand(1, 1, 1, 10)
print(rand_4d_tensor)
sqz_rand_4d_tensor = torch.squeeze(rand_4d_tensor)
print(sqz_rand_4d_tensor)
