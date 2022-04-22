"""
Python implementation of piecewise approximation of an arbitrary sigmoid-like 
(S-curve) function. 

In v3 going away from those approximation objects. Approximation functions are
represented by m and b (lists of scalars)

another thing I want to do is edit out all g functions (which in the paper I call h)
and say this code is only for the positive domain.
"""
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

def generate_approximation(f_true, df, m0=1, line_segs = 8):
        m = [m0*2**(-i) for i in range(line_segs)]
        d = [fsolve(lambda x: df(x) - m_, 0.01)[0] for m_ in m]
        b = [lambda x, d_=_, m_ = __: (f_true(d_) + f_true(x) - m_*(x + d_))/2 for _, __ in zip(d, m)] # list of functions
        g = [lambda x, d_=_, m_=__, b_=___: (f_true(x) + f_true(d_) - (m_*d_ + b_(x))) for _,__,___ in zip(d, m, b)]
        # G = lambda x: min([g_(x) for g_ in g])
        c = [0]
        for i in range(len(g) - 1):
            c.append((f_true(d[i])-f_true(d[i+1])-m[i]*d[i]+m[i+1]*d[i+1])/(m[i+1] - m[i]))
        yc = [g_(c_) for g_, c_ in zip(g, c)]
        deltas = [yc_ - f_true(c_) for yc_, c_ in zip(yc, c)]
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
        return m, bs

def f(x, m, b, g, asymp):
    x_abs = x
    if(x_abs < 0):
        x_abs *= -1
    y = min([m_*x + b_ for m_, b_ in zip(m, b)])
    if(y > asymp):
        y = asymp
    if(x < 0):
        y = g(y)
    return y

def get_approximator(m, b, g, asymp):
    def rtrn(x):
        return f(x, m, b, g, asymp)
    return rtrn

def generate_d(m, df):
    d = [fsolve(lambda x: df(x) - m_, 0.01)[0] for m_ in m]
    return d

def generate_c(m, b):
    c = [0]*len(m)
    for i in range(len(m) - 1):
        c[i+1] = (b[i+1]-b[i])/(m[i]+m[i+1])
    return c
    

def get_asymp_point(m, b, asymp):
    return (asymp - b[-1])/m[-1]

def get_max_dif(m, b, asymp, f_true, df_true):
    # may be the wrong g function but it doesn't matter
    f_approx = get_approximator(m, b, lambda x: -1*x, asymp)
    
    d = generate_d(m, df_true)
    c = generate_c(m, b)
    
    epsilon = .00001
    points = c + d + [get_asymp_point(m,b,asymp)]
    max_dif = max([abs(f_true(point) - f_approx(point)) for point in points])
    max_points = []
    for point in points:
        if(max_dif - abs(f_true(point) - f_approx(point)) <= epsilon):
            max_points.append(point)
    return max_dif, max_points

# in LabVIEW I'm using 16 bit ints as fixed points (1 sign) (7 major) (8 minor)
def scale_for_fixed_point(x):
    return round((x + 128)/256*(65536) - 32768)
    
def inverse_scale_for_fixed_point(x):
    return (x + 32768)/65536*256 - 128

if __name__ == '__main__':
    from math import exp, atan, pi
    from numpy import tanh, cosh
    
    make_plots = False
    make_tables = True
    if(make_plots):
        plt.rcParams.update({'image.cmap': 'viridis'})
        cc = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
         'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
         'Century Schoolbook L',  'Utopia', 'ITC Bookman', 'Bookman',
         'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
        plt.rcParams.update({'font.family':'serif'})
        plt.rcParams.update({'font.size': 10})
        plt.rcParams.update({'mathtext.fontset': 'custom'})
        plt.rcParams.update({'mathtext.rm': 'serif'})
        plt.rcParams.update({'mathtext.it': 'serif:italic'})
        plt.rcParams.update({'mathtext.bf': 'serif:bold'})
        
        plt.close("all")
        savfig = False
        plot_neg = False
        plot_true = True
        
        names = ['sigmoid','arctan','tanh']
        f_trues = [lambda x: 1/(1 + exp(-1*x)), atan, tanh]
        dfs = [lambda x: exp(-1*x)/(1+exp(-1*x))**2, lambda x: 1/(1 + x**2), lambda x: 1/cosh(x)**2]
        gs = [lambda x: 1-x, lambda x: -1*x, lambda x: -1*x]
        m0s = [.25,1,1]
        line_segs = [4,10,4] # find these later
        asymps = [1,pi/2,1]
        # y_mins = [0,-1*(pi/2 + .2),-1.2]
        y_mins = [.5,0,0]
        y_maxs = [1.05, 1.1*(pi/2) ,1.1]
        x_maxs = [8,16,4]
        xticks = [(0,4,8),(0,4,8,12,16),(0,4)]
        yticks = [(.5,1),(0,1),(0,1)]
        for name,f_true,df,g,m0,line_seg,y_min,y_max,x_max,asymp,xtick,ytick in \
                zip(names,f_trues,dfs,gs,m0s,line_segs,y_mins,y_maxs,x_maxs,asymps,xticks,yticks):
            m, b = generate_approximation(f_true,df,m0,line_segs=line_seg)
            f_approx = get_approximator(m, b, g, asymp)
            x_min = 0
            if(plot_neg):
                x_min = -1*x_max
            num = 1000
            x = np.linspace(x_min, x_max, num)
            y = np.array([f_approx(x_) for x_ in x])
            plt.figure(figsize = (3, 3))
            plt.plot(x, y, color = 'k',linewidth=1)
            plt.ylim((y_min, y_max))
            plt.xlim((x_min,x_max))
            plt.xticks(xtick)
            plt.yticks(ytick)
            if(plot_true):
                y_true = np.array([f_true(x_) for x_ in x])
                plt.plot(x, y_true, color = 'r', alpha=0.5,linewidth=1)
            if(savfig):    
                plt.savefig("./plots/making figures/%s.png"%(name), dpi=800)
                plt.savefig("./plots/making figures/%s.svg"%(name))
            plt.show()
            plt.close()
    if(make_tables):
        bs = []
        names = ['sigmoid','arctan','tanh']
        f_trues = [lambda x: 1/(1 + exp(-1*x)), atan, tanh]
        dfs = [lambda x: exp(-1*x)/(1+exp(-1*x))**2, lambda x: 1/(1 + x**2), lambda x: 1/cosh(x)**2]
        gs = [lambda x: 1-x, lambda x: -1*x, lambda x: -1*x]
        m0s = [.25,1,1]
        line_segs = [4,10,4] # find these later
        m_powers = [-2,0,0]
        asymps = [1,pi/2,1]
        max_dif = []
        for name,f_true,df,g,m0,line_seg,m_power,asymp in \
                zip(names,f_trues,dfs,gs,m0s,line_segs,m_powers,asymps):
            print("Table for $\%s approximation"%(name))
            m, b = generate_approximation(f_true,df,m0,line_segs=line_seg)
            bs.append(b)
            c = generate_c(m, b)
            prt = '\hline\n $m$ & $c$ & $b$ \\\ \n \hline \n'
            for m_, c_, b_ in zip(m, c, b):
                prt += ('$2^{{{}}}$ & {} & {} \\\ \n \hline \n'.format(m_power, round(c_,5), round(b_,5)))
                m_power -= 1;
            print(prt)
            max_dif.append(get_max_dif(m, b, asymp, f_true, df))
        
        for name, m in zip(names, max_dif):
            print(name + ": ")
            print("Delta: " + str(m[0]))
            print("argmax: ")
            for s in m[1]:
                print(round(s, 5))
        # latex for b
        b_array = np.zeros((10,3))
        b_array[2:6,0] = np.array(bs[0])
        b_array[:4,1] = np.array(bs[2])
        b_array[:10,2] = np.array(bs[1])
        print("Combined table: \n")
        prt = '\hline\n $m$ & $\sigma$ & $\tanh$ & $\arctan$ \\\ \n \hline \n'
        m_power = 0
        for i in range(10):
            e = ['-','-','-']
            e[0] = str(round(b_array[i,0],5)) if b_array[i,0] != 0 else '-'
            e[1] = str(round(b_array[i,1],5)) if b_array[i,1] != 0 else '-'
            e[2] = str(round(b_array[i,2],5)) if b_array[i,2] != 0 else '-'
            prt += '$2^{{{}}}$ & {} & {} & {} \\\ \n \hline \n'.format(m_power, e[0], e[1], e[2])
            m_power -= 1
        print(prt)
    
    f_true = lambda x: 1/(1 + exp(-1*x))
    df = lambda x: exp(-1*x)/(1+exp(-1*x))**2
    g = lambda x: 1-x
    m0 = .25
    line_segs = 4
    asymp = 1
    m, b = generate_approximation(f_true,df,m0,line_segs=line_seg)
    f_approx = get_approximator(m, b, g, asymp)
    