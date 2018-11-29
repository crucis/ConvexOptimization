from models.optimizers import QuasiNewtonAlgorithm
from functions import functionObj, rosenbrock
from scipy.optimize import minimize
import autograd.numpy as np

x_0 = np.array([2,-2])#np.zeros(2)
f_x = rosenbrock
print('-----------Scipy - QuasiNewtown=BFGS-----------')

f_x_obj = functionObj(f_x)

x_min = minimize(f_x_obj, x_0, method='BFGS')
x_min = f_x_obj.best_x
f_min = f_x_obj.best_f
print('X: ', x_min)
print('F: ', f_min)
print('Function evals: %d'%(f_x_obj.nevals))

print('-----------QuasiNewtonAlgorithm with BFGS-----------')
f_x_obj = functionObj(f_x)

opt = QuasiNewtonAlgorithm(f_x_obj, x_0, formula='BFGS', xtol=1e-6)
opt.find_min()
x_min = f_x_obj.best_x
f_min = f_x_obj.best_f
print('X: ', x_min)
print('F: ', f_min)
print('Function evals: %d'%(f_x_obj.nevals))

print('-----------QuasiNewtonAlgorithm with DFP-----------')
f_x_obj = functionObj(f_x)

opt = QuasiNewtonAlgorithm(f_x_obj, x_0, formula='DFP', xtol=1e-6)
opt.find_min()
x_min = f_x_obj.best_x
f_min = f_x_obj.best_f
print('X: ', x_min)
print('F: ', f_min)
print('Function evals: %d'%(f_x_obj.nevals))