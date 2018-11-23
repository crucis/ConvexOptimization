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

from functions import functionObj, functionObj_multiDim


def run_exercise(func, opt, formula=None, line_search=None, seed=42, epsilon=1e-6, maxIter=1e3, plot_charts=True):
    initial_x_names = [
        '[2 -2]T',
        '[-2 2]T',
        '[-2 -2]T'
    ]

    np.random.seed(seed) # forces repeatability
    x1 = np.array([2, -2], dtype=np.float64)
    x2 = np.array([-2, 2], dtype=np.float64)
    x3 = np.array([-2, -2], dtype=np.float64)


    f_x1 = functionObj(func)
    f_x2 = functionObj(func)
    f_x3 = functionObj(func)

    all_fx = [f_x1, f_x2, f_x3]
    all_x = [x1, x2, x3]
    timings = []

    timings.append(time.process_time())
    for fx, initial_x in zip(all_fx, all_x):
        if line_search is not None:
            ls = line_search(fx, initial_x)
            opt(func=fx, x_0=initial_x, line_search_optimizer=ls, xtol=epsilon, maxIter=maxIter).find_min()
        elif formula is None:
            opt(func=fx, x_0=initial_x, xtol=epsilon, maxIter=maxIter).find_min()
        else:
            opt(func=fx, x_0=initial_x, formula=formula, xtol=epsilon, maxIter=maxIter).find_min()
        timings.append(time.process_time())
        
    timings = list(map(operator.sub, timings[1:], timings[:-1]))

    df = create_df(initial_x_names, all_fx, timings)

    if plot_charts == True:
        opt_name = opt.__name__
        _plot_charts(df, opt_name)

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


def create_df(initial_x_names, all_fx, timings):
    # create dataframe
    methods = ['all_best_x', 'all_best_f', 'best_x', 'best_f', 'fevals','grad_evals', 'nevals', 'all_evals', 'all_x']
    
    dict_fx = {x_name: {method: getattr(fx, method) for method in methods}\
               for x_name, fx in zip(initial_x_names, all_fx)}
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

def plot_contour(f_x_obj, region = ([-40, 40], [-40, 40]), mask=None, optimizers = None, names=None, title='Countor Plot'):
    x1_region = region[0]
    x2_region = region[1]

    x1 = np.linspace(*x1_region, num=1000)
    x2 = np.linspace(*x2_region, num=1000)

    xx, yy = np.meshgrid(x1, x2)

    z = f_x_obj([xx, yy], save_eval=False)
    if mask is not None:
        z[z > mask] = 0
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(12)
    ax.set_xlabel(r'$x_0$')
    ax.set_ylabel(r'$x_1$')
    img = ax.contour(xx, yy, z, cmap = cm.viridis, linestyles = "solid")
    ax.clabel(img, inline=1, fontsize=10)
    if optimizers is not None:
        for i, optimizer in enumerate(optimizers):
            ax.plot(optimizer[:, 0], optimizer[:, 1], 'x--', label=names[i])
    ax.set_title(title)
    if optimizers is not None:
        plt.legend()
    ax.set_xlim(tuple(x1_region))
    ax.set_ylim(tuple(x2_region))
    plt.show()