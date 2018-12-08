import sys
if '../..' not in sys.path:
    sys.path.append('../..')

import time
import autograd.numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

from functions import functionObj
from models.utils import UnconstrainProblem


def run_exercise(func, eqc, iqc, optimizers, initial_x, mu=20, line_search=None, seed=42, epsilon=1e-6, maxIter=1e3, plot_charts=True):
    opt_names = []

    np.random.seed(seed) # forces repeatability
    optimizers += [minimize]
    all_fx = [functionObj(func, eqc=eqc, iqc=iqc) for _ in optimizers]
    timings = []

    timings.append(time.process_time())
    for fx, opt in zip(all_fx, optimizers):
        if type(opt) is tuple:
            opt, line_search = opt
        opt_names += [opt.__name__ if line_search is None else opt.__name__ +' + '+line_search.__name__]
        try:
            if line_search is not None:
                UnconstrainProblem(func=fx, x_0=initial_x, opt=opt, line_search_optimizer=line_search, xtol=epsilon, maxIter=maxIter).find_min()
            elif opt is minimize:
                x0 = initial_x
                while fx.niq/fx.smooth_log_constant > epsilon:
                    res = minimize(fun=fx, x0=x0)
                    if fx._has_eqc:
                        x0 = fx.best_z
                    else:
                        x0 = fx.best_x
                    fx.smooth_log_constant *= mu
                fx.grad_evals = res.njev + res.nhev
            else:
                UnconstrainProblem(func=fx, x_0=initial_x, opt=opt, xtol=epsilon, maxIter=maxIter).find_min()
        except Exception as e:
            print(opt.__name__+" didn't converge. "+repr(e))
        line_search = None
        timings.append(time.process_time())
        
    timings = list(map(operator.sub, timings[1:], timings[:-1]))

    df = create_df(opt_names, all_fx, timings)

    if plot_charts == True:
        opt_name = opt.__name__
        _plot_charts(df, opt_name)

    return df

def create_df(opt_names, all_fx, timings):
    # create dataframe
    methods = ['all_best_x', 'all_best_f', 'best_x', 'best_f', 'fevals','grad_evals', 'nevals', 'all_evals', 'all_x']
    
    dict_fx = {x_name: {method: getattr(fx, method) for method in methods}\
               for x_name, fx in zip(opt_names, all_fx)}
    df = pd.DataFrame(dict_fx).T
    df['best_x'] = df['best_x'].map(lambda x: np.round(x, 7))

    df['run_time (s)'] = timings
    
    return df


def _plot_charts(df, opt_name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(opt_name)
    plt.xlabel('f evals')
    plt.ylabel('$f(x)$')
    #i = 4
    for row_name, row in df.iterrows():
        plt.semilogy(row['all_best_f'], label=row_name)#, marker=i)
        #i = (i + 1) % 11
    plt.legend()
    plt.show()
