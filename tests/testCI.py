from models import functionObj
from models.optimizers import CubicInterpolation
import autograd.numpy as np

f_x = lambda x: x**2 - 4*x + 4
f_x_obj = functionObj(f_x)

opt = CubicInterpolation(f_x_obj, np.random.randn(3), xtol = 1e-6, maxIter=2e10)

print(opt.find_min())
print(f_x_obj.fevals)