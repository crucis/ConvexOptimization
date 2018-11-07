from functions import functionObj_multiDim
from models.optimizers import GaussNewtonMethod, InexactLineSearch
import autograd.numpy as np

f1 = lambda x: x[1]**2 - x[2]
f2 = lambda x: x[2] + x[0] * x[1]
f3 = lambda x: x[0] - 0.4 * x[2]

f_x = lambda x: np.array([f1(x), f2(x), f3(x)], dtype=np.float64)

x = np.array([1, 0.5, 0.2], dtype = np.float64)

f_x_obj = functionObj_multiDim(f_x)

line_search_opt = InexactLineSearch(f_x_obj, x)
opt = GaussNewtonMethod(f_x_obj, x, line_search_opt)


x_min = opt.find_min()
f_min = f_x_obj(x_min, save_eval=False)
print('X: %.9f, %.9f , %.9f \nF_x: %.9f, %.9f, %.9f'%(
    x_min[0], x_min[1], x_min[2], f_min[0], f_min[1], f_min[2]))
print('Function evals: %d'%(f_x_obj.fevals))