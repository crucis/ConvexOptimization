import autograd.numpy as np
from scipy.linalg import null_space
from autograd import grad, jacobian
from copy import copy


class functionObj:
    def __init__(self, func, eqc=None, iqc=None):    
        self.func = func
        self._func_with_no_constraints = func
        self._has_eqc = False
        self._has_iqc = False
        self.fevals = 0
        self.grad_evals = 0
        self.best_x = np.inf
        self.best_f = np.inf
        self.all_best_f = []
        self.all_best_x = []
        self.all_evals = []
        self.all_x = []
        self._grad = grad(self.func)
        if (eqc is not None) or (iqc is not None):
            self.add_constraints(equality=eqc, inequality=iqc)


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


    @ property
    def nevals(self):
        return self.fevals + self.grad_evals

    def grad(self, x, save_eval = True):
        if type(x) == int:
            x = float(x)
        elif hasattr(x, '__iter__'):
            x = np.array(x, dtype = np.float64)
        if not save_eval:
            return self._grad(x)
        result = self._grad(x)
        self.grad_evals = self.grad_evals + 1
        return result


    def _update_params(self, x, result):
        self.fevals = self.fevals + 1
        
        # Autograd ArrayBox behaves differently from numpy, that fixes it.
        result_copy = result if not hasattr(result, '_value') else result._value
        if hasattr(x, '__iter__'):
            if hasattr(x[0], '__iter__'):
                x = list(map(lambda x: list(map(lambda x: x if not hasattr(x, '_value') else x._value, x)), x))
            else:
                x = list(map(lambda x: x if not hasattr(x, '_value') else x._value, x))
        x_copy = x if not hasattr(x, '_value') else x._value

        assert np.any(np.isnan(result_copy)) == False, "X out of domain"
        if self._has_eqc:
            x_copy = np.squeeze(
                self._null_space_feasible_matrix @ np.reshape(x_copy, (-1,1)) + self._feasible_vector)

        self.all_evals += [result_copy]
        self.all_x += [x_copy]

        found_best = np.all(result_copy <= self.best_f)

        if found_best:
            self.all_best_x += [x_copy]
            self.all_best_f += [result_copy]
            self.best_x = x_copy
            self.best_f = result_copy
        return result


    def add_constraints(self, equality=None, inequality=None):
        if equality is not None:
            self._has_eqc = True
            self._feasible_matrix = equality[0]
            self._feasible_solution = equality[1]
            self._feasible_vector = self.find_feasable_solution()
            self._null_space_feasible_matrix = self.find_null_space_feasable_matrix()
            func = copy(self.func)
            self.func = lambda x: np.squeeze(func(
                    np.dot(self._null_space_feasible_matrix, x[:, np.newaxis]) + self._feasible_vector))
            self._grad = grad(self.func)
        #if inequality is not None:
            
        if (equality is None) and (inequality is None):
            raise ValueError("Constraints must be passed to be added.")
        return None


    def find_feasable_solution(self):
        x, _, _, _ = np.linalg.lstsq(self._feasible_matrix, self._feasible_solution)
        return x


    def find_null_space_feasable_matrix(self):
        return null_space(self._feasible_matrix)


    def remove_constraints(self):
        self.func = self._func_with_no_constraints
        return None
        


class functionObj_multiDim(functionObj):
    def __init__(self, func):
        super().__init__(func)
        self._grad = jacobian(self.func)


    def _update_params(self, x, result):
        self.fevals = self.fevals + 1
        
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

        self.all_evals += [[result_copy]]
        self.all_x += [[x_copy]]
        found_best = np.all(result_copy <= self.best_f)

        if found_best:
            self.best_x = x_copy
            self.best_f = result_copy
        return result