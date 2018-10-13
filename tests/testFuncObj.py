from autograd import grad
from functions import functionObj

f = lambda x: x**2 - 4*x

func = functionObj(f)

print(func(-2), func.fevals)

print(func.grad(-2), func.fevals)

grad_func = grad(func)

print(grad_func(-2.0), func.fevals)