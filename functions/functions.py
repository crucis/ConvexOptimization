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


def exercise1116():
    A = np.array([[1,2,1,2], [1,1,2,4]], dtype=np.float64)
    b = np.array([[3], [5]], dtype=np.float64)
    c = np.array([1, 1.5, 1, 1], dtype=np.float64)
    bound = [lambda x: -x[0],lambda x:  -x[1],lambda x:  -x[2],lambda x:  -x[3]]
    f_x = lambda x: np.dot(c, x)
    return f_x, (A,b), bound


def question42():
    f_x = lambda x: np.linalg.norm(x[0:2] - x[2:4])**2
    f1_x = lambda x: -(-x[:2] @ np.array([[0.25, 0], [0, 1]], dtype=np.float64) @ x[:2] + x[:2] @ np.array([0.5, 0], dtype=np.float64) + 0.75)
    f2_x = lambda x: -(-0.125 * (x[2:] @ np.array([[5, 3], [3, 5]], dtype=np.float64) @ x[2:]) + x[2:] @ np.array([5.5, 6.5], dtype=np.float64) - 17.5)
    bound = [f1_x, f2_x]
    return f_x, bound


def question43():
    Q = np.array([[4,0,0], [0,1,-1],[0,-1,1]], dtype=np.float64)
    c = np.array([-8,-6,-6], dtype=np.float64)
    f_x = lambda x: 0.5 * x.T @ Q @ x + x.T @ c
    A = np.array([[1,1,1]], dtype=np.float64)
    b = np.array([[3]], dtype=np.float64)
    bound = [lambda x: -x[0],lambda x:  -x[1],lambda x:  -x[2]]
    return f_x, (A,b), bound


def question44():
    F_0 = np.array([[0.50, 0.55, 0.33, 2.38],
                    [0.55, 0.18, -1.18, -0.40],
                    [0.33, -1.18, -0.94, 1.46],
                    [2.38, -0.40, 1.46, 0.17]], dtype=np.float64)
    F_1 = np.array([[5.19, 1.54, 1.56, -2.80], 
                    [1.54, 2.2, 0.39, -2.5], 
                    [1.56, 0.39, 4.43, 1.77], 
                    [-2.8, -2.5, 1.77, 4.06]], dtype=np.float64)
    F_2 = np.array([[-1.11, 0, -2.12, 0.38], 
                    [0, 1.91, -0.25, -0.58], 
                    [-2.12, -0.25, -1.49, 1.45], 
                    [0.38, -0.58, 1.45, 0.63]], dtype=np.float64)
    F_3 = np.array([[2.69, -2.24, -0.21, -0.74], 
                    [-2.24, 1.77, 1.16, -2.01], 
                    [-0.21, 1.16, -1.82, -2.79], 
                    [-0.74, -2.01, -2.79, -2.22]], dtype=np.float64)
    F_4 = np.array([[0.58, -2.19, 1.69, 1.28], 
                    [-2.19, -0.05, -0.01, 0.91], 
                    [1.69, -0.01, 2.56, 2.14], 
                    [1.28, 0.91, 2.14, -0.75]], dtype=np.float64)

    all_F = np.array([F_0, F_1, F_2, F_3, F_4])
    c = np.array([1, 0, 2, -1], dtype=np.float64)

    f_x = lambda x: np.dot(c.T, x)
    ineq = [lambda x: -(H + np.sum([x[i]*F for i, F in enumerate(all_F[1:])], axis=0)) for H in all_F]
    return f_x, ineq


def question45():
    f_x = lambda x: 100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 + 90*(x[2]**2 - x[3])**2 + (x[2]**2 - 1)**2 + 10.1*((x[1]-1)**2+(x[3]-1)**2) + 19.8*(x[1]-1)*(x[3]-1)
    bound = [lambda x: np.abs(x[0])-10,lambda x:  np.abs(x[1])-10,lambda x:  np.abs(x[2])-10,lambda x:  np.abs(x[3])-10]
    return f_x, bound
