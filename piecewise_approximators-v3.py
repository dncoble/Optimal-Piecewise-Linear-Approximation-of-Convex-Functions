""" piecewise_approximators- v1.py and v2.py can be found at
dncoble/LSTM-implementation-on-FPGA (right now not published). There's a few
things I'd like to do in v3.
1. Make the piecewise_approximator.f function look more like a bitwise shift
algorithm
2. Create some support for provided m, b, c values
3. Abandon 'method 1'/local minimum method
4. remove other code associated with FPGA, ML training not associated with this
project.

"""

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
    
class piecewise_approximator:
    
    # this is a mess. i'm sorry.
    def generate(self, m0 = 1, line_segs = 8):
        m0 = 2**(-m0)
        m = [m0*2**(-i) for i in range(line_segs)]
        d = [fsolve(lambda x: self.df(x) - m_, 0.01)[0] for m_ in m]
        b = [lambda x, d_=_, m_ = __: (self.func(d_) + self.func(x) - m_*(x + d_))/2 for _, __ in zip(d, m)] # list of functions
        g = [lambda x, d_=_, m_=__, b_=___: (self.func(x) + self.func(d_) - (m_*d_ + b_(x))) for _,__,___ in zip(d, m, b)]
        # G = lambda x: min([g_(x) for g_ in g])
        c = [0]
        for i in range(len(g) - 1):
            c.append((self.func(d[i])-self.func(d[i+1])-m[i]*d[i]+m[i+1]*d[i+1])/(m[i+1] - m[i]))
        yc = [g_(c_) for g_, c_ in zip(g, c)]
        deltas = [yc_ - self.func(c_) for yc_, c_ in zip(yc, c)]
        delta_index = deltas.index(max(deltas))
        cs = [0]*line_segs # list of constants (returned)
        bs = [0]*line_segs
        cs[delta_index] = c[delta_index]
        bs[delta_index] = b[delta_index](cs[delta_index])
        for i in range(delta_index - 1, 0, -1): # what to do if an m is skipped?
            c_next = fsolve(lambda x: g[i](x) - (m[i+1]*x + bs[i+1]), cs[i+1])[0]
            cs[i] = c_next
            bs[i] = b[i](c_next)
        bs[0] = b[0](cs[1])
        for i in range(delta_index + 1, line_segs, 1):
            c_next = fsolve(lambda x: g[i](x) - (m[i-1]*x + bs[i-1]), cs[i-1])[0]
            cs[i] = c_next
            bs[i] = b[i](c_next)
        return bs, cs, d
    
    def __init__(self, f, df, g=lambda x: -1*x, asmp=1, name=None, \
                 line_segs=8, m0=1):
        self.m0 = m0 # m0 is the initial bitshift. 2^(-m0) is the initial slope
        self.func = f
        self.df = df
        self.g = g
        self.asmp = asmp
        self.name = name
        self.b, self.c, self.d = self.generate(m0, line_segs)
    
    # this assumes that each m is used. optimally i guess we would have a binary
    # search. certainly this isn't faster in python but i don't think they 
    # have functionality for changing the exponent of a float number
    def f(self, x):
        x_abs = x
        if(x_abs < 0):
            x_abs *= -1
        mx = x_abs*2**(-1*self.m0)
        index = 0
        for i in self.c[1:]:
            if x_abs >= i:
                index += 1
        mx *= 2**(-1*index)
        y = mx + self.b[index]
        
        if(y > self.asmp):
            y = self.asmp
        if(x < 0):
            y = self.g(y)
        return y
    
    def default_df(self, x):
        x_abs = x
        if(x_abs < 0):
            x_abs *= -1
        index = -1
        for i in self.c:
            if x_abs > i:
                index += 1
        dy = self.m[index]
        if(dy > self.asmp):
            y = 0
        if(x < 0):
            y = -1*y # not strictly true, i should multiply by derivative g
        return y
    
    # or, the final c
    def get_asmp_point(self):
        return (self.asmp - self.b[-1])/2**(-(self.m0 + len(self.c) - 1))
    
    # for all possible, use points = c + d + [get_asmp_point]
    # next make it also return location(s) of max point(s)
    def get_delta(self):
        max_dif = 0
        max_point = 0
        for point in self.c + self.d + [self.get_asmp_point()]:
            if(abs(self.func(point) - self.f(point)) > max_dif):
                max_dif = abs(self.func(point) - self.f(point))
                max_point = point
        return max_dif, max_point
    
    #see v2 for some work on making this tensorflow-acceptable
    def get_f(self):
        return self.f
    
    def test_continuity(self, epsilon = 0.0001):
        rtrn = True
        for i in range(1, len(self.c) - 1):
            if(abs(self.m[i-1]*self.c[i]+self.b[i-1] - \
                   (self.m[i]*self.c[i]+self.b[i])) > epsilon):
                rtrn = False
        return rtrn
    
    def get_values(self, latex=True, print_it = False):
        if(not latex):
            if(print_it):
                print('{}\n{}\n{}'.format(self.m, self.c, self.b))
            return '{}\n{}\n{}'.format(self.m, self.c, self.b)
        rtrn = '\hline\n m & c & b \\\ \n \hline \n'
        for m_, c_, b_ in zip(self.m, self.c, self.b):
            rtrn += ('{} & {} & {} \\\ \n \hline \n'.format(m_, c_, b_))
        if(print_it):
            print(rtrn)
        return rtrn
    
    def plot(self, x_max = 16, y_min = 0, y_max = 1, plot_true = False,\
            plot_neg = False, savfig = False, savpath = None):
        x_min = 0
        if(plot_neg):
            x_min = -1*x_max
        num = 1000
        x = np.linspace(x_min, x_max, num)
        y = np.array([self.f(x_) for x_ in x])
        plt.figure(figsize = (15, 4))
        plt.plot(x, y, color = 'k')
        if(plot_true):
            y_true = np.array([self.func(x_) for x_ in x])
            plt.plot(x, y_true, color = 'r', alpha=0.5)
        if(savfig):    
            plt.savefig(savpath, dpi=800)
        plt.show()
        plt.close()

if __name__ == '__main__':
    savfig=False
    #%% sigmoid function
    from math import exp
    f = lambda x: 1/(1 + exp(-1*x))
    df = lambda x: exp(-1*x)/(1+exp(-1*x))**2
    f_bar = piecewise_approximator(f, df, g = lambda x: 1-x, m0=2)
    f_bar.plot(y_max = 1.2, plot_neg = True, plot_true = True, savfig=savfig, savpath="./plots/sig2.png")
    print("sigmoid, method 2 max dif: " + str(f_bar.get_delta()))
    #%% arctan function
    from math import atan, pi
    f = atan
    df = lambda x: 1/(1 + x**2)
    f_bar = piecewise_approximator(f, df, g = lambda x: -1*x, m0=0, asmp=pi/2, line_segs = 12)
    f_bar.plot(y_min = -1*(pi/2 + .2),y_max = pi/2 + .2, plot_neg = True, savfig=savfig, savpath="./plots/atan2.png")
    print("arctan, method 2 max dif: " + str(f_bar.get_delta()))
    #%% tanh function
    from numpy import tanh, cosh
    f = tanh
    df = lambda x: 1/cosh(x)**2
    f_bar = piecewise_approximator(f, df, g = lambda x: -1*x, m0=0)
    f_bar.plot(x_max = 8, y_min = -1.2 ,y_max = 1.2, plot_neg = True, savfig=savfig, savpath="./plots/tanh2.png")
    print("tanh, method 2 max dif: " + str(f_bar.get_delta()))