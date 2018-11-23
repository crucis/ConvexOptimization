from models.optimizers import FletcherReevesAlgorithm, BacktrackingLineSearch
from functions import functionObj
from scipy.optimize import fmin_cg
import autograd.numpy as np

x_0 = np.ones(2)
f_x = lambda x: 100 * (x[1] - x[0]**2) **2 + (1-x[0])**2

print('-----------Non-linear Conjugate from Scipy-----------')

f_x_obj = functionObj(f_x)

x_min, f_min, _, _, _, = fmin_cg(f_x_obj, x_0, full_output=True)
x_min = f_x_obj.best_x
f_min = f_x_obj.best_f
print('X: ', x_min)
print('F: ', f_min)
print('Function evals: %d'%(f_x_obj.fevals))

print('-----------ConjugateDescentAlgorithm-----------')
f_x_obj = functionObj(f_x)
line_search = BacktrackingLineSearch(f_x_obj, x_0)
opt = FletcherReevesAlgorithm(f_x_obj,x_0, line_search, 1e3, xtol=1e-6)
opt.find_min()
x_min = f_x_obj.best_x
f_min = f_x_obj.best_f
print('X: ', x_min)
print('F: ', f_min)
print('Function evals: %d'%(f_x_obj.fevals))