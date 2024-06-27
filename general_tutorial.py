import torch
import numpy as np

### Learning about datatyes and stuff before making the CNN ###
# 2D array
data = [[1, 2],[3, 4]]

# converting the 2D array into a tensor
x_data = torch.tensor(data)

# 
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# ok cool

# shape will be a tupe of tensor dimentions i.e. (rows, columns)
shape = (2,3)
rand_tensor = torch.rand(shape)
# .ones make it all ones (used in ReLU or ArgMax with conditional selection, maybe?)
ones_tensor = torch.ones(shape)
# .zeros make it all zeros
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
# Cool!

# Oh no GPU
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
# Checking the tensor's location
print(f"Device tensor is stored on: {tensor.device}")

tensor = torch.ones(4, 4)
# Remeniscent of pandas (kinda)
print(f"First row: {tensor[0]}")
# Colon: "select all elements along this axis."
print(f"First column: {tensor[:, 0]}")
# Ellipsis: For more complex indexing
print(f"Last column: {tensor[..., -1]}")

# <More complex indexing example dimensions=4>
tensor_4d = torch.ones(2, 3, 4, 5)

# Using colon
print(tensor_4d[:, :, :, 0])  # Selects the first element in the last dimension across all other dimensions

# Using ellipsis
print(tensor_4d[..., 0])  # Equivalent to the above
# </More complex indexing example>

# Making all of the first column zeros
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

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Turns out you can "bridge" NumPy arrays and Tensors

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# Changing one affects the other
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor conversion
n = np.ones(5)
t = torch.from_numpy(n)

# Changing one affects the other (another demonstartion from the otehr end)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")