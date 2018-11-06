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
                              SteepestDescentAlgorithm

def run_exercise(func, opt, line_search, seed=42, epsilon=1e-6):
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
    SteepestDescentAlgorithm(f_x1, x1, line_search_opt1).find_min()
    timings.append(time.process_time())
    SteepestDescentAlgorithm(f_x2, x2, line_search_opt2).find_min()
    timings.append(time.process_time())
    SteepestDescentAlgorithm(f_x3, x3, line_search_opt3).find_min()
    timings.append(time.process_time())
    SteepestDescentAlgorithm(f_x4, x4, line_search_opt4).find_min()
    timings.append(time.process_time())

    timings = list(map(operator.sub, timings[1:], timings[:-1]))

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