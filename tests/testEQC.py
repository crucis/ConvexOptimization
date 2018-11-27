from models.optimizers import QuasiNewtonAlgorithm
from functions import functionObj
import numpy as np


f_x = lambda x: x[0] + 1.5*x[1] + x[2] + x[3]
f_x_obj = functionObj(f_x)
A = np.array([[1,2,1,2], [1,1,2,4]], dtype=np.float64)
b = np.array([[3], [5]], dtype=np.float64)

f_x_obj.add_constraints((A, b))
QuasiNewtonAlgorithm(f_x_obj, np.array([[0],[1]], dtype=np.float64)).find_min()

x_min = f_x_obj.best_x._value
f_min = f_x_obj.best_f._value
print('X: ', x_min)
print('F: ', f_min)
print('Function evals: %d'%(f_x_obj.fevals))