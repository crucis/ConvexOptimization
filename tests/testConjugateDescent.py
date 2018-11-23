from models.optimizers import ConjugateGradientAlgorithm
from functions import functionObj, exercise61
from scipy.optimize import fmin_cg
import autograd.numpy as np

x_0 = np.zeros(16)
f_x = exercise61

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

opt = ConjugateGradientAlgorithm(f_x_obj,x_0,  1e3, xtol=1e-6)
opt.find_min()
x_min = f_x_obj.best_x._value
f_min = f_x_obj.best_f._value
print('X: ', x_min)
print('F: ', f_min)
print('Function evals: %d'%(f_x_obj.fevals))