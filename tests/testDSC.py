from models import functionObj
from models.optimizers import DaviesSwannCampey
import autograd.numpy as np

f_x = lambda x: x**2 - 4*x + 4
f_x_obj = functionObj(f_x)

x_0 = np.random.randn(1)*100
opt = DaviesSwannCampey(f_x_obj)#, 
""" x_0 = x_0,
                        initial_increment = 0.1*x_0, 
                        scaling_constant = 0.1, 
                        xtol = 1e-6, 
                        maxIter=2e10)
"""
x_min = opt.find_min()
print('X: %.9f \nF_x: %.9f'%(x_min, f_x_obj(x_min)))
print('Function evals: %d'%(f_x_obj.fevals - 1))
