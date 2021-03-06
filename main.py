import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(12)
torch.set_num_interop_threads(12)

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

torch.manual_seed(1)

# INITING MATRIX
# torch.tensor(data) creates a torch.Tensor object with the given data.
V_data = [1., 2., 3.]
V = torch.tensor(V_data)
print(V)
print(V[0])
# Get a Python number from it
print(V[0].item())

# Creates a matrix
M_data = [[1., 2., 3.], [4., 5., 6]]
M = torch.tensor(M_data)
print(M)

# Random matrix 3x4x5
x = torch.randn((3, 4, 5))
print(x)

# OPERATIONS
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
z = x + y
print(z)

tStart = time.time()
for i in range(100000):
  t0 = torch.randn((100, 100))
  t1 = torch.randn((100, 100))
  t2 = torch.randn((100, 100))

  for j in range(10):
    t2 += t0 * t1

  # print(t2)

print(time.time() - tStart)
print(torch.get_num_interop_threads())
print(torch.device('cpu'),)
