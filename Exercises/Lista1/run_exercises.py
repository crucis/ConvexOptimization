import sys
if '../..' not in sys.path:
    sys.path.append('../..')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import time
from copy import copy
import operator

from scipy.optimize import brute
from functions import order5_polynomial, logarithmic, sinoid, order4_polynomial
from functions import functionObj
from models.optimizers import DichotomousSearch, FibonacciSearch, GoldenSectionSearch, \
                                QuadraticInterpolationSearch, CubicInterpolation, DaviesSwannCampey, \
                                BacktrackingLineSearch, InexactLineSearch
                                
def run_exercise(func, f_string, interval, plot_func = True, seed = 32, epsilon = 1e-5, textpos = (3,5)):
    all_fx_names = ['Brute Force', 
                'Dichotomous Search', 
                'Fibonacci Search', 
                'Golden-Section Search', 
                'Quadratic Interpolation Method', 
                'Cubic interpolation Method', 
                'Davies, Swann and Campey Algorithm', 
                'Backtracking Line Search']

    np.random.seed(seed) # forces repeatability
    
    # objects that log all info during minimization
    f_x = functionObj(func) 
    f_x_DS = functionObj(func) 
    f_x_FBS = functionObj(func) 
    f_x_GSS = functionObj(func) 
    f_x_QIM = functionObj(func) 
    f_x_CIM = functionObj(func) 
    f_x_DSC = functionObj(func) 
    f_x_BLS = functionObj(func) 
    all_fx = [f_x, f_x_DS, f_x_FBS, f_x_GSS, f_x_QIM, f_x_CIM, f_x_DSC, f_x_BLS]
    
    # Brute Force
    start_time = time.process_time()
    min_brute = brute(f_x, (tuple(interval), ), full_output = True)
    brute_time = time.process_time() - start_time
    # Plot function if wanted
    if plot_func == True:
        x = np.linspace(interval[0], interval[1], 100)
        plt.plot(x, f_x(x, save_eval = False), label = f_string)
        plt.annotate('min_x: %.6f'%(min_brute[0][0]) +'\nmin_fx: %.6f'%(min_brute[1]), 
                     xy=textpos, xycoords='axes pixels') 
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.legend()
        plt.show()

    timings = []
    # Minimizations
    timings.append(time.process_time())
    DichotomousSearch(f_x_DS, epsilon = epsilon/10, interval = interval, xtol = epsilon).find_min()
    timings.append(time.process_time())
    FibonacciSearch(f_x_FBS, interval = interval, xtol = epsilon).find_min()
    timings.append(time.process_time())

    GoldenSectionSearch(f_x_GSS, interval = interval, xtol = epsilon).find_min()
    timings.append(time.process_time())
    QuadraticInterpolationSearch(f_x_QIM, interval = interval, xtol = epsilon).find_min()
    timings.append(time.process_time())
    CubicInterpolation(f_x_CIM, interval = interval, xtol = epsilon).find_min()
    timings.append(time.process_time())
    DaviesSwannCampey(f_x_DSC, x_0 = np.random.uniform(interval[0], interval[1], size=1), 
                      interval = interval, xtol = epsilon).find_min()
    timings.append(time.process_time())

    BacktrackingLineSearch(f_x_BLS, initial_x = np.random.uniform(interval[0], interval[1], size=1), 
                           interval = interval, xtol = epsilon).find_min()
    timings.append(time.process_time())
    
    timings = list(map(operator.sub, timings[1:], timings[:-1]))
    timings = [brute_time] + timings
    # Create dataframe
    methods = ['best_x', 'best_f', 'fevals', 'all_evals', 'all_x']
    dict_fx = {fx_name: {method: getattr(fx, method) for method in methods}\
               for fx_name, fx in zip(all_fx_names, all_fx)}
    df = pd.DataFrame(dict_fx).T
    df['best_f'] = df['best_f'].map(lambda x: x if not hasattr(x, '__iter__') else x[0])
    df['best_x'] = df['best_x'].map(lambda x: x if not hasattr(x, '__iter__') else x[0])
    df['all_evals'] = df['all_evals'].map(lambda x: np.array(x) if not hasattr(x[0], '__iter__') \
                                          else np.array(x).flatten())
    df['all_x'] = df['all_x'].map(lambda x: np.array(x) if not hasattr(x[0], '__iter__') \
                                  else np.array(x).flatten())
    df['run_time (s)'] = timings
    return df


def plot_surface():
    func = functionObj(order4_polynomial)
    x1_region = [-np.pi, np.pi]
    x2_region = [-np.pi, np.pi]

    x1 = np.linspace(*x1_region)
    x2 = np.linspace(*x2_region)

    xx, yy = np.meshgrid(x1, x2)

    z = func([xx, yy], save_eval=False)
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    ax = Axes3D(fig)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$f(\mathbf{x})$')
    img = ax.plot_surface(xx, yy, z, cmap = cm.viridis, alpha = 0.9)
    ax.contour(xx, yy, z, cmap = cm.viridis, linestyles = "solid", offset = -40)
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=None,
                        hspace=None)
    plt.colorbar(img)
    #plt.show()
    plt.savefig('demo.png', orietantion = 'portrait', pbox_inches = 'tight')

def plot_contour(result_fletcher = None, result_back = None):
    func = functionObj(order4_polynomial)
    x1_region = [-np.pi, np.pi]
    x2_region = [-np.pi, np.pi]

    x1 = np.linspace(*x1_region)
    x2 = np.linspace(*x2_region)

    xx, yy = np.meshgrid(x1, x2)

    z = func([xx, yy], save_eval=False)
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    img = ax.contour(xx, yy, z, cmap = cm.viridis, linestyles = "solid")
    ax.clabel(img, inline=1, fontsize=10)
    if result_fletcher is not None:
        ax.plot(result_fletcher[0], result_fletcher[1], 'r+', label="Fletcher's inexact line search solution.")
    if result_back is not None:
        ax.plot(result_back[0], result_back[1], 'ko', label="Backtracking inexact line search solution.")
    ax.set_title('Contour plot')
    if (result_back is not None) or (result_fletcher is not None):
        plt.legend()
    plt.show()

def plot_func_alpha(x_0, d_0, alpha, fletcher_alpha, fletcher_f, back_alpha, back_f):
    func = functionObj(order4_polynomial)
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$f(\mathbf{x}_0+\alpha \mathbf{d}_0)$')
    z = x_0 + alpha[:, np.newaxis]@np.array(d_0)[:,np.newaxis].T
    f = func(z.T, save_eval=False)
    ax.plot(alpha, f, label=r"$f(\mathbf{x}_0 + \alpha\mathbf{d}_0)$")
    ax.plot(fletcher_alpha, fletcher_f, 'r+', label="Fletcher's inexact line search solution.")
    ax.plot(back_alpha, back_f, 'ko', label="Backtracking inexact line search solution.")
    plt.legend()
    plt.show()