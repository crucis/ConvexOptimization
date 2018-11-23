class optimizer:
    def __init__(self, func, x_0=None, maxIter = 1e6, interval = [-5, 5], xtol = 1e-6, ftol = 1e-6):
        self.objectiveFunction = func
        self.maxIter = int(maxIter)
        self.interval = interval
        self.xtol = xtol
        self.ftol = ftol
        self.x_0 = x_0

    def find_min(self):
        pass