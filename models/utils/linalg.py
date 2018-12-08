import autograd.numpy as np

def is_pos_def(A):
    #A = remove_arraybox(A)
    #A = np.array(A, dtype=np.float64)
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
        except TypeError:
            return np.all(np.linalg.eigvals(A) > 0)
    else:
        return False

def remove_arraybox(x):
    while hasattr(x, '_value'):
        x = x._value
    if np.size(np.shape(x)) >= 1:
        for i in range(x.shape[0]):
            while hasattr(x[i], '_value'):
                x[i] =  x[i]._value
            if np.size(np.shape(x)) >= 2:
                for j in range(x.shape[1]):
                    while hasattr(x[i][j], '_value'):
                        x[i][j] =  x[i][j]._value
    return x