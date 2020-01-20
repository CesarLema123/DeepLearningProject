import torch
import numpy as np

##################################################################################################
# Tensor, Function and Autograd
##################################################################################################

'''
Setting "tensor.requires_grad = True" tracks all operations on the tensor. The tensor method ".backwards()" computes the gradients automatically. The tensor attribute ".grad" collects the computed values.

The tensor method ".detach()" return new tensor (with same storage as original tensor) detachde from current graph. Detaches from the computation history.

The tensor method ".no_grad()"; Context-manager that disabled gradient calculation, useful when sure that ".backwards()" will not be called. Useful for computational efficiency.

The "Function" class and "Tensor " class are interconnected and build up an acyclic graph, that encodes computation history. The tensor attribute ".grad_fn" references a function that created it.
'''

x = torch.ones(2,2, requires_grad = True)                   # Initializes tensor and specifies to track operations on it
x0 = x + 2



print(x)
print(x.grad_fn)
print(x0)




##################################################################################################
# Gradients
##################################################################################################
