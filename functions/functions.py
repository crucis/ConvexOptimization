import autograd.numpy as np


def order5_polynomial(x):
    assert len(x) == 1, 'x must be a one-dimensional variable'
    x = np.array(x, dtype=np.float64)
    return -5*x**5 + 4*x**4 - 12*x**3 + 11*x**2 - 2*x +1

def logarithmic(x):
    assert len(x) == 1, 'x must be a one-dimensional variable'
    x = np.array(x, dtype=np.float64)
    return np.log(x-2)**2 + np.log(10-x)**2 - x**0.2

def sinoid(x):
    assert len(x) == 1, 'x must be a one-dimensional variable'
    x = np.array(x, dtype=np.float64)
    return -3*x*np.sin(0.75*x) + np.exp(-2*x)

def order4_polynomial(x):
    assert len(x) == 2, 'x must be a 2-dimensional variable'
    x = np.array(x, dtype=np.float64)
    return 0.7*x[0]**4 - 8*x[0]**2 + 6*x[1]**2 + np.cos(x[0]*x[1]) - 8*x[0]

def exercise57(x):
    assert len(x) == 2, 'x must be a 2-dimensional variable'
    x = np.array(x, dtype=np.float64)
    return (x[0]**2 + x[1]**2 - 1) **2 + (x[0] + x[1] - 1)**2

def exercise520(x):
    assert len(x) == 4, 'x must be a 4-dimensional variable'
    x = np.array(x, dtype=np.float64)
    return (x[0] + 10*x[1]) **2 + 5*(x[2] - x[3]) **2 + (x[1] - 2* x[2]) ** 4 + 100 * (x[0] - x[3])**4

def exercise520_gauss(x):
    assert len(x) == 4, 'x must be a 4-dimensional variable'
    x = np.array(x, dtype=np.float64)
    f1 = lambda x: x[0] + 10*x[1]
    f2 = lambda x: np.sqrt(5) * (x[2] - x[3])
    f3 = lambda x: (x[1] - 2*x[2]) ** 2
    f4 = lambda x: 10 * (x[0] - x[3]) ** 2
    return np.array([f1(x), f2(x), f3(x), f4(x)], dtype=np.float64)

def exercise61(x):
    assert len(x) == 16, 'x must be a 16-dimensional variable'
    Q1 = np.array([[12, 8, 7, 6], [8, 12, 8, 7], [7, 8, 12, 8], [6, 7, 8, 12]], dtype=np.float64)
    Q2 = np.array([[3, 2, 1, 0], [2, 3, 2, 1], [1, 2, 3, 2], [0, 1, 2, 3]], dtype=np.float64)
    Q3 = np.array([[2, 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1], [0, 0, 1, 2]], dtype=np.float64)
    Q4 = np.eye(4)
    Q_l1 = np.hstack([Q1, Q2, Q3, Q4])#, [Q2, Q1, Q2, Q3], [Q3, Q2, Q1, Q2], [Q4, Q3, Q2, Q1]], axis=0)
    Q_l2 = np.hstack([Q2, Q1, Q2, Q3])
    Q_l3 = np.hstack([Q3, Q2, Q1, Q2])
    Q_l4 = np.hstack([Q4, Q3, Q2, Q1])
    Q = np.vstack([Q_l1, Q_l2, Q_l3, Q_l4])

    b = -np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float64)

    x_0 = np.ones_like(b)
    f_x = lambda x: 0.5 * np.dot(np.dot(x.T, Q), x) + np.dot(b.T, x)
    return f_x(x)


def rosenbrock(x):
    assert len(x) == 2, 'Rosenbrock function made for two-dimensional variables'
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def exercise54(x):
    assert len(x) == 2, 'x must be a 2-dimensional variable'
    return 5*x[0]**2 - 9 *x[0]*x[1] + 4.075*x[1]**2 + x[0]