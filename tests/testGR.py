from models import functionObj
from models.optimizers import GoldenSectionSearch

f_x = lambda x: x**2 - 4*x + 4
f_x_obj = functionObj(f_x)

opt = GoldenSectionSearch(f_x_obj, xtol = 1e-6, maxIter=2e10)

print(opt.find_min())
print(f_x_obj.fevals)