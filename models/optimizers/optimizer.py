class optimizer:
    def __init__(self, func, maxIter = 1e6, interval = [-1e6, 1e6], xtol = 1e-6, ftol = 1e-6):
        self.objectiveFunction = func
        self.maxIter = int(maxIter)
        self.interval = interval
        self.xtol = xtol
        self.ftol = ftol
    
    def find_min(self):
        pass