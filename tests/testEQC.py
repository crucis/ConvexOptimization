from models.optimizers import SteepestDescentAlgorithm, BacktrackingLineSearch
from models.utils import UnconstrainProblem
from functions import functionObj
from scipy import optimize
from autograd import numpy as np

A = np.array([[1,2,1,2], [1,1,2,4]], dtype=np.float64)
b = np.array([[3], [5]], dtype=np.float64)
c = np.array([1, 1.5, 1, 1], dtype=np.float64)
bound = [lambda x: -x[0],lambda x:  -x[1],lambda x:  -x[2],lambda x:  -x[3]]
f_x = lambda x: np.dot(c, x)
print('----------ScipySimplex----------')

res = optimize.linprog(c, A_eq = A, b_eq=b)
print(res)

print('----------QuasiNewton----------')
f_x_obj = functionObj(f_x)
f_x_obj.add_constraints(equality=(A, b), inequality=bound)
x_0 = np.array([0, 0], dtype=np.float64)
line_search = BacktrackingLineSearch(f_x_obj, x_0)

UnconstrainProblem(f_x_obj, x_0, SteepestDescentAlgorithm, line_search_optimizer=line_search).find_min()
x_min = np.array(f_x_obj.best_x)
print('x: ', x_min)
f_min = f_x_obj.best_f
print('F: ', f_min)
print('Function evals: %d'%(f_x_obj.nevals))