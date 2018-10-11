import autograd.numpy as np
from copy import copy

class functionObj:
    def __init__(self, func):    
        self.func = func
        self.fevals = 0
        self.best_x = np.inf
        self.best_f = np.inf
        self.all_evals = []
        self.all_x = []

    def __call__(self, x, save_eval = True):
        if not save_eval:
            return self.func(x)
        self.fevals = self.fevals + 1
        result = self.func(x)
        
        # Autograd ArrayBox behaves differently from numpy, that fixes it.
        if type(result) == np.numpy_boxes.ArrayBox:
            result_copy = copy(result._value)
        else:
            result_copy = copy(result)
        if type(x) == np.numpy_boxes.ArrayBox:
            x_copy = x._value
        else:
            x_copy = x
        
        assert np.isnan(result_copy).all() == False, "X out of domain"
        self.all_evals += [result_copy]
        self.all_x += [x_copy]
        if (result_copy < self.best_f).any():
            self.best_x = x_copy
            self.best_f = result_copy
        return result

    def reset_fevals(self):
        self.__init__(self.func)
