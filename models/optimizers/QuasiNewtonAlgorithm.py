from .optimizer import optimizer
import autograd.numpy as np
from copy import copy

def _quasi_newton_algorithm(x, objectiveFunction, formula, xtol, maxIter):
    # step 1
    rho = 0.1
    sigma = 0.7
    tau = 0.1
    chi = 0.75
    M_hat = 600
    epsilon_2 = 10**(-10)
    S = np.eye(len(x))
    f = objectiveFunction(x)
    gk = objectiveFunction.grad(x)
    f_00 = copy(f)
    delta_f = copy(f)

    for _ in range(maxIter): 
        # step 2 line search
        direction_vector = - np.dot(S, gk)
        alpha_L = 0
        alpha_U = np.inf
        f_L = f
        grad_fL = np.dot(objectiveFunction.grad(x + alpha_L * direction_vector).T, direction_vector)
        # estimate alpha_0
        if abs(grad_fL) > epsilon_2:
            alpha_0 = -2*delta_f/grad_fL
        else:
            alpha_0 = 1
        if (alpha_0 <= 0) or (alpha_0 > 1):
            alpha_0 = 1
        # step 3
        for _ in range(maxIter):
            if alpha_0 < xtol:
                break
            delta_k = alpha_0*direction_vector
            if np.square(np.linalg.norm(delta_k)) < xtol:
                break
            f = objectiveFunction(x+delta_k)

            # step 4 interpolation
            if (f > f_L + rho*(alpha_0 - alpha_L)*grad_fL) and (abs(f_L - f) > epsilon_2):# and (objectiveFunction.fevals < M_hat):
                if alpha_0 < alpha_U:
                    alpha_U = alpha_0
                
                alpha_tilde = _compute_a0tilde(alpha_0, f, f_L, grad_fL, alpha_L)
                alpha_tilde_L = alpha_L + tau*(alpha_U - alpha_L)
                if alpha_tilde < alpha_tilde_L:
                    alpha_tilde = alpha_tilde_L
                alpha_tilde_U = alpha_U - tau*(alpha_U - alpha_L)
                if alpha_tilde > alpha_tilde_U:
                    alpha_tilde = alpha_tilde_U
                alpha_0 = alpha_tilde
            else:
                # step 5
                grad_f = np.dot(objectiveFunction.grad(x+alpha_0*direction_vector).T, direction_vector)

                # step 6 extrapolation
                if (grad_f < sigma * grad_fL) and (abs(f_L - f) > epsilon_2):# and (~np.isclose(grad_fL - grad_f, 0)): 
                    delta_alpha_0 = (alpha_0 - alpha_L)*grad_f/(grad_fL - grad_f + 1e-9)
                    if delta_alpha_0 <= 0:
                        alpha_tilde = 2*alpha_0
                    else:
                        alpha_tilde = alpha_0 + delta_alpha_0

                    alpha_tilde_U = alpha_0 + chi*(alpha_U - alpha_0)
                    if alpha_tilde > alpha_tilde_U:
                        alpha_tilde = alpha_tilde_U
                    alpha_L = alpha_0
                    alpha_0 = alpha_tilde
                    f_L = f
                    grad_fL = grad_f

                else:
                    # step 7
                    x = x + delta_k
                    delta_f = f_00 - f
                    if (np.sqrt(np.linalg.norm(delta_k)) < xtol) and (abs(delta_f) < xtol):# or (objectiveFunction.fevals >= M_hat):
                        return x
                    f_00 = f
                    break
        # step 8
        gk_new = objectiveFunction.grad(x)
        gamma_k = gk_new - gk
        gk = copy(gk_new)
        D = np.dot(delta_k.T, gamma_k)
        if D <= 0:
            S = np.eye(x.shape[0])
        else:
            S = _compute_Sk(formula, S, delta_k, gamma_k)

    return x


def _compute_a0tilde(a0, f0, f_L, grad_f_L, alphaL):
    numerator = (a0 - alphaL)**2 * grad_f_L
    denominator = 2*(f_L - f0 + (a0 - alphaL) * grad_f_L)
    return alphaL + numerator/denominator


def _compute_Sk(method, S, delta_k, gamma_k):
    if method == 'DFP':
        S = _compute_Sk_DFP(S, delta_k, gamma_k)
    elif method == 'BFGS':
        S = _compute_Sk_BFGS(S, delta_k, gamma_k)
    return S


def _compute_Sk_DFP(S, delta_k, gamma_k):
    second_term = np.dot(delta_k[:, np.newaxis], delta_k[:, np.newaxis].T)/np.dot(delta_k.T, delta_k)
    F = np.dot(S, gamma_k[:, np.newaxis])
    third_term = np.dot(F, F.T)/np.dot(gamma_k[:, np.newaxis].T, F)
    S = S + second_term - third_term
    return S


def _compute_Sk_BFGS(S, delta_k, gamma_k):
    first_term_left = 1 + np.dot(np.dot(gamma_k[:, np.newaxis].T, S), gamma_k)/np.dot(gamma_k[:, np.newaxis].T, delta_k)
    first_term_right = np.dot(delta_k[:, np.newaxis], delta_k[:, np.newaxis].T)/np.dot(gamma_k, delta_k)
    second_term = (np.dot(np.dot(delta_k[:, np.newaxis], gamma_k[:, np.newaxis].T), S) + \
                np.dot(np.dot(S, gamma_k[:, np.newaxis]), delta_k[:, np.newaxis].T))/np.dot(gamma_k[:, np.newaxis].T, delta_k[:, np.newaxis])
    return S + first_term_left*first_term_right - second_term


class QuasiNewtonAlgorithm(optimizer):
    def __init__(self,
                 func,
                 x_0,
                 formula = 'BFGS',
                 maxIter = 1e3,
                 xtol = 1e-6,
                 ftol = 1e-6):
        self.objectiveFunction = func
        self.x_k = x_0
        possible_formulas = ['DFP', 'BFGS']
        assert formula in possible_formulas, "Formula must be one of " + str(possible_formulas)
        self.formula = formula
        super().__init__(func = func, maxIter = maxIter, xtol = xtol, ftol = ftol)

    def find_min(self):
            return _quasi_newton_algorithm(self.x_k, self.objectiveFunction, self.formula, self.xtol, self.maxIter)


    