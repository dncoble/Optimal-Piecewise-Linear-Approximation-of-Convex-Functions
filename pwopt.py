"""
Author: Daniel Coble

A python implementation of the algorithm used to generate optimal piecewise
linear approximations of convex functions proposed by Imamoto and Tang.

Imamoto, A., & Tang, B. (2008, October). Optimal piecewise linear approximation
of convex functions. In Proceedings of the world congress on engineering and 
computer science (pp. 1191-1194).

Args:
    f: float -> float function to be a approximated.
    df: float -> float derivative of f.
    alpha_0: lower bound of approximated domain.
    alpha_N: upper bound of approximated domain.
    N: positive integer number of linear pieces in PW approximation.
    delta: Desired accuracy in the errors of the iterative algorithm. If None
    it will be set to 1/1000th maximum error.
    convex: set True if the function is convex. Set False if the function is
    concave (negative convex).
    set_m: If a list, generate_approximation optimizes the approximation with
    the given slopes. If None, the standard algorithm varying slopes will be 
    applied.
"""
class PW_approx:
    
    def __init__(self, m, c, t, alpha, convex):
        self.m = m; self.c = c; self.t = t; self.alpha = alpha;
        self.convex = convex
        
    def __call__(self, x):
        if(self.convex):
            return max([m_i*x+c_i for m_i, c_i in zip(self.m, self.c)])
        return min([m_i*x+c_i for m_i, c_i in zip(self.m, self.c)])
        
    
def generate_approximation(f_, df_, alpha_0, alpha_N, N, convex=True, delta=None, set_m=None):
    if(not convex):
        f = lambda x: -1*f_(x)
        df = lambda x: -1*df_(x)
    else:
        f = f_
        df = df_
    
    Delta=1;
    t = [alpha_0 + (1/N)*(i+.5)*(alpha_N-alpha_0) for i in range(N)]
    
    finished = False
    prevepsilon = None
    prevt = None
    # for l in range(7):
    while(not finished):
        alpha = [alpha_0] + \
            [(f(t[i-1])-f(t[i])+df(t[i])*t[i]-df(t[i-1])*t[i-1])/(df(t[i])-df(t[i-1])) for i in range(1,N)]\
            + [alpha_N]
        g = [lambda x, t_i=t_i: df(t_i)*(x-t_i) + f(t_i) for t_i in t]
        epsilon = [g[i](alpha[i]) - f(alpha[i]) for i in range(N)]+[g[N-1](alpha[N])]
        abs_epsilon = [abs(epsilon_i) for epsilon_i in epsilon]
        if(delta is None):
            if(max(abs_epsilon)/min(abs_epsilon) - 1 < max(abs_epsilon)*.001):
                finished = True
        else:
            if(max(abs(epsilon))/min(abs(epsilon)) - 1 < delta):
                finished = True
        if(not finished):
            if(prevepsilon is not None and max(abs_epsilon) > max([abs(pe) for pe in prevepsilon])):
                Delta /= 2; epsilon = prevepsilon; t = prevt
            else:
                prevt = t
            d = [Delta*(epsilon[i+1] - epsilon[i])/(epsilon[i+1]/(alpha[i+1]-t[i])+epsilon[i]/(t[i]-alpha[i])) for i in range(N)]
            t = [t[i]+d[i] for i in range(N)]
            prevepsilon = epsilon;
        print(max(abs_epsilon))
    epsilon = .5*max(epsilon)
    m = [df(t_i) for t_i in t]
    c = [-1*df(t_i)*t_i+f(t_i)-epsilon for t_i in t]
    if(not convex):
        m = [-1*m_i for m_i in m]
        c = [-1*c_i for c_i in c]
    g = PW_approx(m, c, t, alpha, convex)
    return g

# example uses and generating gif for README
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from math import sin, cos
    g = generate_approximation(sin, cos, 0, math.pi, 7, convex=False)
    
    x = np.linspace(0, math.pi, 1000, endpoint=True)
    y_f = np.array([sin(x_) for x_ in x])
    y_g = np.array([g(x_) for x_ in x])
    
    plt.figure(figsize=(7, 3))
    plt.plot(x, y_f, label="sin(x)")
    plt.plot(x, y_g, label="PW approx")
    plt.legend()
    plt.tight_layout()
    
    

    
    
    