import sys
if '../..' not in sys.path:
    sys.path.append('../..')

import time
import autograd.numpy as np
import operator
import pandas as pd

from functions import functionObj
from models.optimizers import InexactLineSearch,\
                              BacktrackingLineSearch,\
                              SteepestDescentAlgorithmm,\
                              GaussNewtonMethod

def run_exercise(func, opt, line_search, seed=42, epsilon=1e-6, plot_charts=True):
    initial_x_names = [
        '[4 4]T',
        '[4 -4]T',
        '[-4 4]T',
        '[-4 -4]T'
    ]

    np.random.seed(seed) # forces repeatability
    x1 = [4, 4]
    x2 = [4, -4]
    x3 = [-4, 4]
    x4 = [-4, 4]


    f_x1 = functionObj(func)
    line_search_opt1 = line_search if line_search is None else line_search(f_x1, x1)
    f_x2 = functionObj(func)
    line_search_opt2 = line_search if line_search is None else line_search(f_x2, x2)
    f_x3 = functionObj(func)
    line_search_opt3 = line_search if line_search is None else line_search(f_x3, x3)
    f_x4 = functionObj(func)
    line_search_opt4 = line_search if line_search is None else line_search(f_x4, x4)


    all_fx = [f_x1, f_x2, f_x3, f_x4]
    timings = []

    timings.append(time.process_time())
    opt(func=f_x1, x_0=x1, line_search_optimizer=line_search_opt1).find_min()
    timings.append(time.process_time())
    opt(func=f_x2, x_0=x2, line_search_optimizer=line_search_opt2).find_min()
    timings.append(time.process_time())
    opt(func=f_x3, x_0=x3, line_search_optimizer=line_search_opt3).find_min()
    timings.append(time.process_time())
    opt(func=f_x4, x_0=x4, line_search_optimizer=line_search_opt4).find_min()
    timings.append(time.process_time())

    timings = list(map(operator.sub, timings[1:], timings[:-1]))

    df = create_df(initial_x_names, all_fx, timings)

    if plot_charts == True:
        line_search_name = 'with '+line_search.__name__ \
            if line_search is not None else 'without line search'
        opt_name = opt.__name__
        _plot_charts(df, opt_name, line_search_name)

    return df


def _plot_charts(df, opt_name, line_search_name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title(opt_name+' '+line_search_name)
    plt.xlabel('f evals')
    plt.ylabel('$f(x)$')
    #i = 4
    for row_name, row in df.iterrows():
        plt.semilogy(row['all_evals'], label=row_name)#, marker=i)
        #i = (i + 1) % 11
    plt.legend()
    plt.show()


def run_exercise520(func, opt, line_search, seed=42, epsilon=1e-6, plot_charts=True):
    initial_x_names = [
        '[-2 -1 1 2]T',
        '[200 -200 100 -100]T'
    ]

    np.random.seed(seed) # forces repeatability
    x1 = np.array([-2, -1, 1, 2], dtype=np.float64)
    x2 = np.array([200, -200, 100, -100], dtype=np.float64)


    f_x1 = functionObj(func)
    line_search_opt1 = line_search if line_search is None else line_search(f_x1, x1)
    f_x2 = functionObj(func)
    line_search_opt2 = line_search if line_search is None else line_search(f_x2, x2)

    all_fx = [f_x1, f_x2]
    timings = []

    timings.append(time.process_time())
    opt(func=f_x1, x_0=x1, line_search_optimizer=line_search_opt1).find_min()
    timings.append(time.process_time())
    opt(func=f_x2, x_0=x2, line_search_optimizer=line_search_opt2).find_min()
    timings.append(time.process_time())

    timings = list(map(operator.sub, timings[1:], timings[:-1]))

    df = create_df(initial_x_names, all_fx, timings)

    if plot_charts == True:
        line_search_name = 'with '+line_search.__name__ \
            if line_search is not None else 'without line search'
        opt_name = opt.__name__
        _plot_charts(df, opt_name, line_search_name)

    return df

def create_df(initial_x_names, all_fx, timings):
    # create dataframe
    methods = ['best_x', 'best_f', 'fevals', 'all_evals', 'all_x']
    
    dict_fx = {x_name: {method: getattr(fx, method) for method in methods}\
               for x_name, fx in zip(initial_x_names, all_fx)}
    df = pd.DataFrame(dict_fx).T
    df['best_f'] = df['best_f']#.map(lambda x: x if not hasattr(x, '__iter__') else x[0])
    df['best_x0'] = df['best_x'].map(lambda x: x if not hasattr(x, '__iter__') else x[0])
    df['best_x1'] = df['best_x'].map(lambda x: x if not hasattr(x, '__iter__') else x[1])

    df['all_evals'] = df['all_evals'].map(lambda x: np.array(x) if not hasattr(x[0], '__iter__') \
                                          else np.array(x).flatten())

    df['all_x'] = df['all_x'].map(lambda x: np.array(x) if not hasattr(x[0], '__iter__') \
                                  else np.array(x).flatten())
    df['all_evals'] = df['all_evals'].map(lambda x: x._value if hasattr(x, '_value') \
                                          else x)
    df['all_x'] = df['all_x'].map(lambda x: x._value if hasattr(x, '_value') \
                                          else x)
    df['run_time (s)'] = timings
    
    return df