from models.optimizers import SteepestDescentAlgorithm, BacktrackingLineSearch
from functions import functionObj
from scipy import optimize
from autograd import numpy as np

A = np.array([[1,2,1,2], [1,1,2,4]], dtype=np.float64)
b = np.array([[3], [5]], dtype=np.float64)
c = np.array([1, 1.5, 1, 1], dtype=np.float64)
f_x = lambda x: np.dot(c, x)
print('----------ScipySimplex----------')

res = optimize.linprog(c, A_eq = A, b_eq=b)
print(res)

print('----------QuasiNewton----------')
f_x_obj = functionObj(f_x)
f_x_obj.add_constraints((A, b))
x_0 = np.array([-1,-1], dtype=np.float64)
line_search = BacktrackingLineSearch(f_x_obj, x_0)

SteepestDescentAlgorithm(f_x_obj, x_0, line_search_optimizer=line_search).find_min()
x0 = np.dot(f_x_obj._null_space_feasible_matrix, x_0[:, np.newaxis]) + f_x_obj._feasible_vector
x_min = np.array(f_x_obj.best_x)
print('x: ', x_min)
f_min = f_x_obj.best_f
print('F: ', f_min)
print('Function evals: %d'%(f_x_obj.nevals))