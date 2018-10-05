import autograd.numpy as np

class functionObj:
    def __init__(self, func):    
        self.func = func
        self.fevals = 0
        self.best_x = np.inf

    def __call__(self, x):
        self.fevals = self.fevasl + 1
        result = self.func(x)
        assert np.isnan(result) == False, "X out of domain"
        return result

    def reset_fevals(self):
        self.fevals = 0
