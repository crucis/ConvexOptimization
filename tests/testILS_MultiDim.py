from functions import functionObj
from models.optimizers import InexactLineSearch
import autograd.numpy as np
from autograd import grad
from copy import copy

f_x = lambda x: x[0]**2 - 4*x[0]*x[1] + 4*x[1]**2
f_x_obj = functionObj(f_x)

x_0 = np.random.randn(2)*100
opt = InexactLineSearch(f_x_obj, x_0)

x_min = opt.find_min()
print('X: %.9f, %.9f \nF_x: %.9f'%(x_min[0], x_min[1], f_x_obj(x_min)))
print('X1-X2 : %.4e'%(x_min[0] - 2* x_min[1]))
print('Function evals: %d'%(f_x_obj.fevals - 1))
