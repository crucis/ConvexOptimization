import autograd.numpy as np

class FilterDesign:
    def __init__(self, passband_edge, 
                  stopband_edge, 
                  passband_error, 
                  passband_gain, 
                  order_length,
                  nb_sampled_frequencies_pass,
                  nb_sampled_frequencies_stop,
                  stopband_gain=None,
                  stopband_attenuation=None,
                  gamma=25,
                  kapa=0):
        if stopband_gain is None:
            if stopband_attenuation is None:
                raise ValueError('Stopband_gain or stopband_attenuation must be passed.')
            self.stopband_gain = 10**(-0.05*stopband_attenuation)
        else:
            self.stopband_gain = stopband_gain

        self.passband_edge = passband_edge
        self.stopband_edge = stopband_edge
        self.passband_error = passband_error
        self.passband_gain = passband_gain 
        self.order_length = order_length
        self.nb_sampled_frequencies_pass = nb_sampled_frequencies_pass
        self.nb_sampled_frequencies_stop = nb_sampled_frequencies_stop
        self.gamma = gamma

        self.Q = get_Q(w_a=self.stopband_edge, w_p=self.passband_edge, order_length=self.order_length, gamma=self.gamma)
        self.b_l = get_bl(order_length=self.order_length, w_p=self.passband_edge)
        self.A = get_A(nb_sampled_frequencies_pass=self.nb_sampled_frequencies_pass,
                        nb_sampled_frequencies_stop=self.nb_sampled_frequencies_stop,
                        passband_edge=self.passband_edge,
                        stopband_edge=self.stopband_edge, 
                        order_length=self.order_length)
        self.b = get_b_constraint(stopband_gain=self.stopband_gain,
                                    nb_sampled_frequencies_pass=self.nb_sampled_frequencies_pass,
                                    nb_sampled_frequencies_stop=self.nb_sampled_frequencies_stop,
                                    passband_error=self.passband_error)
        self.kapa = kapa
        self.freq_response = lambda w, x: complex(np.dot(x.T, get_cl(w, order_length)), 
                                                    -np.dot(x.T, get_sl(w, order_length)))
        self.f_x = lambda x: np.dot(np.dot(x.T, self.Q), x) - 2*np.dot(self.b_l, x) + self.kapa
        self.iqc = lambda x: np.dot(self.A, x) - self.b

class minMaxFilterDesign:
    def __init__(self):
        pass





def get_Q(w_a, w_p, order_length,  gamma=25):
    array_size = int( (order_length + 2)/2)
    rng = np.arange(1, array_size+1)
    f1 = lambda i: w_p/2 + np.sin(2*(i-1)*w_p)/(4*(i-1)) if i != 1 else w_p
    f2 = lambda i, j: np.sin((i-j)*w_p)/(2*(i-j)) + np.sin((i+j-2)*w_p)/(2*(i+j-2))
    Q_L1 = np.array([[f1(i) if i==j else f2(i,j) for i in rng] for j in rng], dtype=np.float64)
    f3 = lambda i: gamma*((np.pi-w_a)/2 - np.sin(2*(i-1)*w_a)/(4*(i-1))) if i != 1 else gamma*(np.pi-w_a)
    f4 = lambda i,j: -gamma/2 * (np.sin((i-j)*w_a)/(i-j) + np.sin((i+j-2)*w_a)/(i+j-2))
    Q_L2 = np.array([[f3(i) if i==j else f4(i,j) for i in rng]for j in rng], dtype=np.float64)
    return Q_L1 + Q_L2


def get_bl(w_p, order_length):
    array_size = int((order_length + 2)/2)
    rng = np.arange(1, array_size+1)
    b = np.array([np.sin((n-1)*w_p)/(n-1) if n != 1 else w_p for n in rng], dtype=np.float64)
    return b


def get_cl(w, order_length):
    rng = np.arange(0, order_length/2 + 1)
    return [np.cos(i*w) for i in rng]


def get_sl(w, order_length):
    rng = np.arange(0, order_length/2 + 1)
    return [np.sin(i*w) for i in rng]

def get_Ap(nb_sampled_frequencies, passband_edge, stopband_edge, order_length):
    sampled_frequencies = np.linspace(passband_edge, stopband_edge, nb_sampled_frequencies)
    App = np.array([get_cl(w, order_length) for w in sampled_frequencies], dtype=np.float64)
    Anp = np.array([-np.array(get_cl(w, order_length)) for w in sampled_frequencies], dtype=np.float64)
    return np.concatenate((App, Anp))


def get_bp(nb_sampled_frequencies, passband_error):
    bpp = np.array([1 + passband_error for _ in range(nb_sampled_frequencies)])
    bnp = np.array([-1 + passband_error for _ in range(nb_sampled_frequencies)])
    bp = np.concatenate((bpp, bnp))
    return bp


def get_b_constraint(stopband_gain, nb_sampled_frequencies_pass, nb_sampled_frequencies_stop, passband_error):
    ba = stopband_gain * np.ones(nb_sampled_frequencies_stop * 2)
    bp = get_bp(nb_sampled_frequencies_pass, passband_error)
    b = np.concatenate((bp, ba))
    return b


def get_A(nb_sampled_frequencies_pass, nb_sampled_frequencies_stop, passband_edge, stopband_edge, order_length):
    Ap = get_Ap(nb_sampled_frequencies_pass, 0, passband_edge, order_length)
    Aa = get_Ap(nb_sampled_frequencies_stop, stopband_edge, np.pi, order_length)
    return np.concatenate((Ap, Aa))

