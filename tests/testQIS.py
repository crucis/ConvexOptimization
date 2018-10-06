from models import functionObj
from models.optimizers import QuadraticInterpolationSearch

f_x = lambda x: x**2 - 4*x + 4
f_x_obj = functionObj(f_x)

opt = QuadraticInterpolationSearch(f_x_obj, xtol = 1e-6, maxIter=2e10)


print('X: %.9f \nF_x: %.9f'%(opt.find_min(), f_x_obj(opt.find_min())))
print('Function evals: %d'%(f_x_obj.fevals - 1))
