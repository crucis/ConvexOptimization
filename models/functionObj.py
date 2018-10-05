import autograd.numpy as np

class functionObj:
    def __init__(func, domain = (-1e28, 1e28)):    
        self.func = func
        self.domain = domain
        self.fevals = 0
        self.best_x = np.inf

    def __call__(x):
        self.fevals = self.fevasl + 1
        result = self.func(x)
        assert np.isnan(result) == False, "X out of domain"
        return result

    def reset_fevals():
        self.fevals = 0
