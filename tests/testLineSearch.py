import numpy as np
from functions import order4_polynomial, functionObj
from models.optimizers import InexactLineSearch, BacktrackingLineSearch

x_0 = np.array([-np.pi, np.pi])
d_0 = np.array([1.0, -1.1])
func = functionObj(order4_polynomial)
item_d_optimizer = InexactLineSearch(func, x_0, d_0)
backtracking_opt = BacktrackingLineSearch(func, x_0, d_0)
alpha_f, f0_f = item_d_optimizer._line_search()
alpha_b, f0_b = backtracking_opt._backtracking_line_search(func.grad(x_0))
print('Inexact Line Search Methods line search step:')
print(' - Fletcher solution\n   ' + 'alpha'+': %.7f\n   '%alpha_f + '.' + 'f: %.7f'%f0_f)
print(' - Backtracking solution\n   '+ 'alpha'+': %.7f\n   '%alpha_b + '.' + 'f: %.7f'%f0_b)

func_f = functionObj(order4_polynomial)
func_b = functionObj(order4_polynomial)
item_d_optimizer = InexactLineSearch(func, x_0, d_0)
#backtracking_opt = BacktrackingLineSearch(func, x_0, d_0)
item_d_optimizer.find_min()
#backtracking_opt.find_min
print('Inexact Line Search Methods for minimization:')
print(' - Fletcher solution\n   ' + 'x_min'+': %s\n   '%func_f.best_x + '.' + 'f_min: %.7f'%func_f.best_f)
#print(' - Backtracking solution\n   '+ 'x_min'+': %s\n   '%func_b.best_x + '.' + 'f_min: %.7f'%func_b.best_f)
