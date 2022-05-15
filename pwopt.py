"""
Author: Daniel Coble

A python implementation of the algorithm used to generate optimal piecewise
linear approximations of convex functions proposed by Imamoto and Tang. Use
generate_approximation to generate a function object which can be called
for the piecewise approximation. 

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
    
    def __init__(self, m, c, t, alpha, epsilon, convex):
        self.m = m; self.c = c; self.t = t; self.alpha = alpha;
        self.epsilon = epsilon; self.convex = convex
        
    def __call__(self, x):
        if(self.convex):
            return max([m_i*x+c_i for m_i, c_i in zip(self.m, self.c)])
        return min([m_i*x+c_i for m_i, c_i in zip(self.m, self.c)])
        
    
def generate_approximation(f, df, alpha_0, alpha_N, N, convex=True, delta=None, set_m=None):
    if(not convex):
        f_ = lambda x: -1*f(x)
        df_ = lambda x: -1*df(x)
        if(set_m is not None):
            set_m = [-1*m_i for m_i in set_m]
    else:
        f_ = f
        df_ = df
    
    finished = False
    
    Delta=1;
    t = [alpha_0 + (1/N)*(i+.5)*(alpha_N-alpha_0) for i in range(N)]
    
    if(set_m is not None): # a bisection search for pivot locations
        #check and throw error
        if(max(set_m) > df_(alpha_N) or min(set_m) < df_(alpha_0)):
            raise Exception("Given slopes cannot be found within bounds.")
        t_upper = [alpha_N for i in range(N)]
        t_lower = [alpha_0 for i in range(N)]
        t = [.5*(t_upper[i] + t_lower[i]) for i in range(N)]
        if(delta is None):
            delta = .00001
        while(not finished):
            m_prop = [df_(t_i) for t_i in t]
            epsilon = [set_m[i] - m_prop[i] for i in range(N)]
            abs_epsilon = [abs(epsilon_i) for epsilon_i in epsilon]
            if(max(abs_epsilon) < delta):
                finished = True
            else:
                t_upper = [t_upper[i] if epsilon[i] > 0 else t[i] for i in range(N)]
                t_lower = [t[i] if epsilon[i] > 0 else t_lower[i] for i in range(N)]
                t = [.5*(t_upper[i] + t_lower[i]) for i in range(N)]
        alpha = [alpha_0] + \
            [(f_(t[i-1])-f_(t[i])+df_(t[i])*t[i]-df_(t[i-1])*t[i-1])/(df_(t[i])-df_(t[i-1])) for i in range(1,N)]\
            + [alpha_N]
    prevepsilon = None
    prevt = None
    while(not finished):
        alpha = [alpha_0] + \
            [(f_(t[i-1])-f_(t[i])+df_(t[i])*t[i]-df_(t[i-1])*t[i-1])/(df_(t[i])-df_(t[i-1])) for i in range(1,N)]\
            + [alpha_N]
        g = [lambda x, t_i=t_i: df_(t_i)*(x-t_i) + f_(t_i) for t_i in t]
        epsilon = [g[i](alpha[i]) - f_(alpha[i]) for i in range(N)]+[g[N-1](alpha[N])]
        abs_epsilon = [abs(epsilon_i) for epsilon_i in epsilon]
        if(delta is None):
            if(max(abs_epsilon)/min(abs_epsilon) - 1 < max(abs_epsilon)*.001):
                finished = True
        else:
            if(max(abs_epsilon)/min(abs_epsilon) - 1 < delta):
                finished = True
        if(not finished):
            if(prevepsilon is not None and max(abs_epsilon) > max([abs(pe) for pe in prevepsilon])):
                Delta /= 2; epsilon = prevepsilon; t = prevt
            else:
                prevt = t
            d = [Delta*(epsilon[i+1] - epsilon[i])/(epsilon[i+1]/(alpha[i+1]-t[i])+epsilon[i]/(t[i]-alpha[i])) for i in range(N)]
            t = [t[i]+d[i] for i in range(N)]
            prevepsilon = epsilon;
    epsilon = .5*min(epsilon)
    m = [df_(t_i) for t_i in t]
    c = [-1*df_(t_i)*t_i+f_(t_i)-epsilon for t_i in t]
    if(not convex):
        m = [-1*m_i for m_i in m]
        c = [-1*c_i for c_i in c]
        epsilon = -1*epsilon
    g = PW_approx(m, c, t, alpha, epsilon, convex)
    return g

# example uses and generating gif for README
if __name__ == '__main__':
    #%% simple use to create a 7-piece approximation of sin
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
    #%% using set_m to generate a sigmoid with only powers of two slope
    from math import exp
    
    f = lambda x: 1/(1+exp(-1*x))
    df = lambda x: exp(-1*x)/(1+exp(-1*x))**2
    set_m = [2**-2, 2**-3, 2**-4, 2**-5]
    alpha_0 = 0; alpha_N=10; N=4
    convex=False
    delta=None
    g = generate_approximation(f, df, 0, 4, 4, set_m=set_m, convex=False)
    x = np.linspace(0, 4, 1000, endpoint=True)
    y_f = np.array([f(x_) for x_ in x])
    y_g = np.array([g(x_) for x_ in x])
    
    plt.figure(figsize=(7, 3))
    plt.plot(x, y_f, label="sigma(x)")
    plt.plot(x, y_g, label="PW approx")
    plt.legend()
    plt.tight_layout()
    #%% animating approximation with increasing amount of pieces
    import matplotlib.animation as animation
    
    fig, ax = plt.subplots(figsize=(5,5))
    x = np.arange(0, math.pi, 0.01)
    y_f = np.array([sin(x_) for x_ in x])
    ax.plot(x, y_f)
    g = generate_approximation(sin, cos, 0, math.pi, 2, convex=False)
    line, = ax.plot(x, np.array([g(x_) for x_ in x]))
    
    def animate(i):
        g = generate_approximation(sin, cos, 0, math.pi, i+2, convex=False)
        line.set_ydata(np.array([g(x_) for x_ in x]))
        ax.text(.25,1, "N=" + str(i+2), backgroundcolor='white',size='x-large')
        ax.text(2.25,1, r'$\varepsilon$={:5.4f}'.format(g.epsilon), backgroundcolor='white',size='x-large')
        return line,
    ani = animation.FuncAnimation(
        fig, animate, frames=18, blit=True
    )
    
    ani.save("increase_line_segs.gif", dpi=500)
    
    
    