import autograd.numpy as np
from autograd import grad
from copy import copy

class functionObj:
    def __init__(self, func):    
        self.func = func
        self.fevals = 0
        self.best_x = np.inf
        self.best_f = np.inf
        self.all_evals = []
        self.all_x = []
        self._grad = grad(self.func)


    def __call__(self, x, save_eval = True):
        if type(x) == int:
            x = float(x)
        elif hasattr(x, '__iter__'):
            x = np.array(x, dtype = np.float64)
            
        if not save_eval:
            return self.func(x)
        result = self.func(x)
        return self._update_params(x, result)


    def reset_fevals(self):
        self.__init__(self.func)


    def grad(self, x, save_eval = True):
        if type(x) == int:
            x = float(x)
        elif hasattr(x, '__iter__'):
            x = np.array(x, dtype = np.float64)
        
        if not save_eval:
            return self._grad(x)
        result = self._grad(x)
        return self._update_params(x, result)


    def _update_params(self, x, result):
        self.fevals = self.fevals + 1
        
        # Autograd ArrayBox behaves differently from numpy, that fixes it.
        if type(result) == np.numpy_boxes.ArrayBox:
            result_copy = copy(result._value if not hasattr(result._value, '__iter__') \
                                                else result._value[0])
        else:
            result_copy = copy(result)
        if type(x) == np.numpy_boxes.ArrayBox:
            x_copy = x._value if not hasattr(x._value, '__iter__') else x._value[0]
        else:
            x_copy = x

        assert np.isnan(result_copy).all() == False, "X out of domain"

        self.all_evals += [result_copy]
        self.all_x += [x_copy]
        if hasattr(result_copy, '__iter__') or hasattr(self.best_f, '__iter__'):
            found_best = (result_copy <= self.best_f).all()
        else:
            found_best = result_copy < self.best_f 

        if found_best:
            self.best_x = x_copy
            self.best_f = result_copy
        return result
