import sys
if '../..' not in sys.path:
    sys.path.append('../..')

import time
import autograd.numpy as np
import operator
import pandas as pd

from functions import functionObj, functionObj_multiDim


def run_exercise(func, opt, seed=42, epsilon=1e-6, plot_charts=True):
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
    timings = []

    timings.append(time.process_time())
    opt(func=f_x1, x_0=x1).find_min()
    timings.append(time.process_time())
    opt(func=f_x2, x_0=x2).find_min()
    timings.append(time.process_time())
    opt(func=f_x3, x_0=x3).find_min()
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
        plt.semilogy(row['all_evals'], label=row_name)#, marker=i)
        #i = (i + 1) % 11
    plt.legend()
    plt.show()


def create_df(initial_x_names, all_fx, timings):
    # create dataframe
    methods = ['best_x', 'best_f', 'fevals','grad_evals', 'nevals', 'all_evals', 'all_x']
    
    dict_fx = {x_name: {method: getattr(fx, method) for method in methods}\
               for x_name, fx in zip(initial_x_names, all_fx)}
    df = pd.DataFrame(dict_fx).T
    df['best_f'] = df['best_f']#.map(lambda x: x if not hasattr(x, '__iter__') else x[0])
    df['best_f'] = df['best_f'].map(lambda x: x if not hasattr(x, '_value') else x._value)
    if hasattr(df.iloc[0]['best_x'], '__iter__'):
        for i in range(len(df.iloc[0]['best_x'])):
            df['best_x'+str(i)] = df['best_x'].map(lambda x: x if not hasattr(x, '__iter__') else x[i])
    else:
        df['best_x'] = df['best_x'].map(lambda x: x if not hasattr(x, '__iter__') else x[0])
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