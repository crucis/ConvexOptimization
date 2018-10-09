from functions import functionObj
from models.optimizers import GoldenSectionSearch

f_x = lambda x: x**2 - 4*x + 4
f_x_obj = functionObj(f_x)

opt = GoldenSectionSearch(f_x_obj, xtol = 1e-6, maxIter=2e10)

x_min = opt.find_min()
print('X: %.9f \nF_x: %.9f'%(x_min, f_x_obj(x_min)))
print('Function evals: %d'%(f_x_obj.fevals - 1))
