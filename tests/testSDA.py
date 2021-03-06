from functions import functionObj
from models.optimizers import SteepestDescentAlgorithm, InexactLineSearch
import autograd.numpy as np

f_x = lambda x: x[0]**2 - 4*x[0]*x[1] + 4*x[1]**2
f_x_obj = functionObj(f_x)

x_0 = np.random.randn(2)*100
line_search_opt = InexactLineSearch(f_x_obj, x_0)
opt = SteepestDescentAlgorithm(f_x_obj, x_0, line_search_opt)

print('------- WITH INEXACT LINE SEARCH -------')
x_min = opt.find_min()
print('X: %.9f, %.9f \nF_x: %.9f'%(x_min[0], x_min[1], f_x_obj(x_min)))
print('X1-X2 : %.4e'%(x_min[0] - 2* x_min[1]))
print('Function evals: %d'%(f_x_obj.fevals - 1))

print('------- WITHOUT LINE SEARCH -------')
f_x_obj.reset_fevals()
opt = SteepestDescentAlgorithm(f_x_obj, x_0)

x_min = opt.find_min()
print('X: %.9f, %.9f \nF_x: %.9f'%(x_min[0], x_min[1], f_x_obj(x_min)))
print('X1-X2 : %.4e'%(x_min[0] - 2* x_min[1]))
print('Function evals: %d'%(f_x_obj.fevals - 1))