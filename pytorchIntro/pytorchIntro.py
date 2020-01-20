import numpy as np
import torch

##################################################################################################
# Different ways of initializing Tesnors; Tensors are similar to matrices except they can run on GPU
##################################################################################################

x0 = torch.empty(5,3)                           # Tensor with uninitialized values
x1 = torch.rand(5,3)                            # Tensor with randomly initialized values
x2 = torch.zeros(5,3,dtype= torch.long)         # Zero Tensor
x3 = torch.tensor([3,4])                        # Tensor from data matrix

'''
print(x0)
print(x1)
print(x2)
print(x3)   '''


##################################################################################################
# Reusing tensors
##################################################################################################

x = torch.zeros(2,2,dtype = torch.long)
x0 = x.new_ones(4,4)                            # new_* methods takes in sizes and resuses old tensors properties
x1 = torch.randn_like(x, dtype = torch.float)   # New tensor reusing shape of input tensor

'''
print(x0)
print(x1)   '''


##################################################################################################
# Operations
##################################################################################################

x = torch.zeros(2,2)
y = torch.ones(2,2)

x0 = x+y                                        # Syntax 1, Binary operation
x1 = torch.add(x,y)                             # Syntax 2, method
result = torch.zeros(2,2)
x2 = torch.add(x,y, out = result)               # Syntax 3, method putting result in an output tensor
x3 = x.new_ones(2,2)
x3.add_(y)                                      # Syntac 4, mutating/in place addition, operations mutating in place is postfixed with an underscore "_"

'''
print(x0)
print(x1)
print(x2)
print(x3)   '''


##################################################################################################
# Tensor Methods
##################################################################################################

x = torch.rand(4,4)

tensorSize = x.size()                           # method returns tensor size
xPrime = x.view(1,16)                           # method returns a resized tensor
indexedValue = x[0][0].item()                          # method return a single element tensor as python datatype

'''
print(tensorSize)
print(xPrime)
print(indexedValue) '''


##################################################################################################
# Numpy bridge
##################################################################################################

x = torch.ones(1,5)                             # Initializing torch tensor
xPrime = x.numpy()                              # Converting to numpy tensor

'''
print(x)
print(xPrime)   '''

x.add_(1)                                       # Tensor and matrix share memory location and mutations are reflected on both objects

'''
print(x)
print(xPrime)   '''


y = np.ones((1,5))                              # Initializing numpy matrix
yPrime = torch.from_numpy(y)                    # Converting to torch tensor

'''
print(y, "\n", yPrime)  '''

np.add(y, 1, out=y)                             # Tensor and matrix share memory location and mutations are reflected on both objects

'''
print(y, "\n", yPrime)  '''


##################################################################################################
# CUDA Tensors, tensor objects can be moved to GPU
##################################################################################################

x = torch.ones(2,2)

if torch.cuda.is_available():                   # Test this code using GPU on Cluster
    device = torch.device("cuda")
    x0 = torch.ones_like(x, device = device)    # New tensor reusing shape of input tensor and directly creates object on GPU
    x1 = x.to(device)                           # Method performs Tensor dtype and/or device conversion returning a copy of self unless self satisfies the desired properties then returns self
    #x2 = x + x0                                 # Can you operate on cpu and cuda tensors together?
    x2 = x0 + x1
    
    print(x)
    print(x0)
    print(x2)

print(x)













