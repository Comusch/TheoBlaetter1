import numpy as np
import matplotlib.pyplot as plt

def v(t,g=9.81,gamma_m=0):
    """Returns dz/dt; """
    if gamma_m==0.:
        return -g*t
    return g/gamma_m * (1-np.exp(gamma_m*t))

ts = np.linspace(0,1,100)
#plot curves for various values of gamma/m
for gamma_m in np.linspace(0,-10,11):
    plt.plot(ts,v(ts,gamma_m=gamma_m),label=r'\$gamma/m=${0}/s'.format(gamma_m))
#set plot parameters
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('v [m/s]')
plt.xlim(ts[0],ts[-1])
plt.show()