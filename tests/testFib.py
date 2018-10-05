from models import functionObj
from models.optimizers import FibonacciSearch

f_x = lambda x: x**2 - 4*x + 4
f_x_obj = functionObj(f_x)

opt = FibonacciSearch(f_x_obj, epsilon = 1e-9, xtol = 1e-6, maxIter=1e6)

print(opt.find_min())
print(f_x_obj.fevals)