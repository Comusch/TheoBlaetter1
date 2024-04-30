import numpy
import matplotlib.pyplot as plt
import numpy as np

print("-------Plot of x(t), v(t) and m(t)---------")

def m(t, g=9.81, alpha= 1, R_0=1, rho=1):
    R = alpha*t*(1/4)+R_0
    print(f"R: {R}")
    V = 4/3*np.pi*R**3
    print(f"V: {V}")
    return rho*V

def v(t, g=-9.81, alpha= 1, R_0 = 1):
    #my own calculated result
    if alpha == 0:
        return g*t
    return 1/4*g*t+R_0*1/alpha*g-(alpha/4*t+R_0)**(-3)*R_0**4*1/alpha*g

def x(t, g=9.81, alpha= 1, R_0=1):
    if alpha == 0:
        return 1/2*g*t**2
    return 1/8*g*t**2+R_0*1/alpha*g*t+1/2*(alpha/4*t+R_0)**(-2)*R_0**4*g*4/alpha**2 - 1/2*R_0**2*g*4/alpha**2

print("-------Plot of m(t)---------")
ts = np.linspace(0, 1.7, 100)

for alpha in np.linspace(0, 10, 11):
    plt.plot(ts, m(ts, alpha = alpha, R_0=1), label=f"alpha: {alpha}")
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('m [kg]')
plt.savefig("Blatt1_m.png")
plt.show()


print("-------Plot of v(t)---------")
ts = np.linspace(0, 1.7, 100)

for alpha in np.linspace(0, 10, 11):
    plt.plot(ts, v(ts, alpha = alpha, R_0=1), label=f"alpha: {alpha}")

plt.legend()
plt.xlabel('t [s]')
plt.ylabel('v [m/s]')
plt.savefig("Blatt1_v.png")
plt.show()


print("-------Plot of x(t)---------")
ts = np.linspace(0, 1.7, 100)

for alpha in np.linspace(0, 10, 11):
    plt.plot(ts, x(ts, alpha = alpha, R_0=1), label=f"alpha: {alpha}")

plt.legend()
plt.xlabel('t [s]')
plt.ylabel('x [m]')
plt.savefig("Blatt1_x.png")
plt.show()
