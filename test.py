
import numpy as np
import torch
from torch.autograd import Variable

dims = 2
uniqWordSize = 3

W1 = Variable(torch.randn(dims, uniqWordSize).float(), requires_grad=True)
print(W1)

oneHotVector = np.zeros(uniqWordSize)
oneHotVector[1] = 1

x = Variable(torch.from_numpy(oneHotVector)).float()
print(x)

a1 = torch.matmul(W1, x)
print(a1)

x = Variable(torch.randn(uniqWordSize).float(), requires_grad=True)
print(x)

a1 = torch.matmul(W1, x)
print(a1)



