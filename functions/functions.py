import autograd.numpy as np



def order5_polynomial(x):
    x = np.array(x, dtype=np.float64)
    return -5*x**5 + 4*x**4 - 12*x**3 + 11*x**2 - 2*x +1

def logarithmic(x):
    x = np.array(x, dtype=np.float64)
    return np.log(x-2)**2 + np.log(10-x)**2 - x**0.2

def sinoid(x):
    x = np.array(x, dtype=np.float64)
    return -3*x*np.sin(0.75*x) + np.exp(-2*x)

def order4_polynomial(x):
    x = np.array(x, dtype=np.float64)
    return 0.7*x[0]**4 - 8*x[0]**2 + 6*x[1]**2 + np.cos(x[0]*x[1]) - 8*x[0]

def exercise57(x):
    x = np.array(x, dtype=np.float64)
    return (x[0]**2 + x[1]**2 - 1) **2 + (x[0] + x[1] - 1)**2

def exercise520(x):
    x = np.array(x, dtype=np.float64)
    return (x[0] + 10*x[1]) **2 + 5*(x[2] - x[3]) **2 + (x[1] - 2* x[2]) ** 4 + 100 * (x[0] - x[3])**4

def exercise520_gauss(x):
    x = np.array(x, dtype=np.float64)
    f1 = x[0] + 10*x[1]
    f2 = np.sqrt(5) * (x[2] - x[3])
    f3 = (x[1] - 2*x[2]) ** 2
    f4 = 10 * (x[0] - x[3]) ** 2
    return np.array([f1(x), f2(x), f3(x), f4(x)], dtype=np.float64)