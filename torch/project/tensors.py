import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")


# Attributs of a Tensor

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape} \n")
print(f"Datatype of tensor: {tensor.dtype} \n")
print(f"Device tensor is stored on: {tensor.device} \n")


# Operations on Tensor

# Move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]} \n")
print(f"First column: {tensor[:, 0]} \n")
print(f"Last column: {tensor[..., -1]} \n")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# Single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place operations
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Bridge with NumPy

# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t} \n")
n = t.numpy()
print(f"n: {n} \n")

t.add_(1)
print(f"t: {t} \n")
print(f"n: {n} \n")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t} \n")
print(f"n: {n} \n")
