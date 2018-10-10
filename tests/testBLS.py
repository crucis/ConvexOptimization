from functions import functionObj
from models.optimizers import BacktrackingLineSearch
import autograd.numpy as np
import matplotlib.pyplot as plt

f_x = lambda x: x**2 - 4*x + 4
f_x_obj = functionObj(f_x)

np.random.seed(42)

x_0 = np.random.randn(1)*100

opt = BacktrackingLineSearch(f_x_obj,
                            initial_x = x_0,
                            alpha = 0.01,
                            beta = 0.5)


x_min = opt.find_min()
print('X: %.9f \nF_x: %.9f'%(x_min, f_x_obj(x_min)))
print('Function evals: %d'%(f_x_obj.fevals - 1))


plt.plot(f_x_obj.all_evals)
plt.show()