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

from functions import functionObj
from models.utils import UnconstrainProblem


def run_exercise(func, eqc, iqc, optimizers, formula=None, line_search=None, seed=42, epsilon=1e-6, maxIter=1e3, plot_charts=True):
    initial_f_names = []

    np.random.seed(seed) # forces repeatability
    initial_x = np.array([0, 0], dtype=np.float64)

    all_fx = [functionObj(func, eqc, iqc) for _ in optimizers]#[f_x1, f_x2, f_x3]
    all_x = [initial_x]
    timings = []

    timings.append(time.process_time())
    for fx, opt in zip(all_fx, optimizers):
        if type(opt) is tuple:
            opt, line_search = opt
        initial_f_names += [opt.__name__]
        try:
            if line_search is not None:
                ls = line_search(fx, initial_x)
                UnconstrainProblem(func=fx, x_0=initial_x, opt=opt, line_search_optimizer=ls, xtol=epsilon, maxIter=maxIter).find_min()
                line_search = None
            elif formula is None:
                UnconstrainProblem(func=fx, x_0=initial_x, opt=opt, xtol=epsilon, maxIter=maxIter).find_min()
            else:
                UnconstrainProblem(func=fx, x_0=initial_x, opt=opt, formula=formula, xtol=epsilon, maxIter=maxIter).find_min()
        except:
            pass
        timings.append(time.process_time())
        
    timings = list(map(operator.sub, timings[1:], timings[:-1]))

    df = create_df(initial_f_names, all_fx, timings)

    if plot_charts == True:
        opt_name = opt.__name__
        _plot_charts(df, opt_name)

    return df

def create_df(initial_f_names, all_fx, timings):
    # create dataframe
    methods = ['all_best_x', 'all_best_f', 'best_x', 'best_f', 'fevals','grad_evals', 'nevals', 'all_evals', 'all_x']
    
    dict_fx = {x_name: {method: getattr(fx, method) for method in methods}\
               for x_name, fx in zip(initial_f_names, all_fx)}
    df = pd.DataFrame(dict_fx).T
    df['best_f'] = df['best_f'].map(lambda x: x if not hasattr(x, '_value') else x._value)
    df['best_x'] = df['best_x'].map(lambda x: x if not hasattr(x, '_value') else x._value)

    if hasattr(df.iloc[0]['best_x'], '_value'):
        for i in range(len(df.iloc[0]['best_x'])):
            df['best_x'+str(i)] = df['best_x'].map(lambda x: x if not hasattr(x, '_value') else x._value[i])
    else:
        df['best_x'] = df['best_x'].map(lambda x: x if not hasattr(x, '_value') else x._value)
    df['all_evals'] = df['all_evals'].map(lambda x: x._value if hasattr(x, '_value') \
                                          else x)
    df['all_x'] = df['all_x'].map(lambda x: x._value if hasattr(x, '_value') \
                                          else x)
    df['all_best_x'] = df['all_best_x'].map(lambda x: x._value if hasattr(x, '_value') \
                                          else x)
    df['all_best_f'] = df['all_best_f'].map(lambda x: x._value if hasattr(x, '_value') \
                                          else x)
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
