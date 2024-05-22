import matplotlib.pyplot as plt
import numpy as np

def v(r, alpha, lambda_v, v_0, r_0):
    v_neu = np.exp(alpha/lambda_v * np.exp(-r*lambda_v)) * (-v_0)/np.exp(alpha/lambda_v * np.exp(-r_0*lambda_v))
    print(f"v_neu: {v_neu}")
    return v_neu

r = np.linspace(0, 10, 100)

for alpha in np.linspace(-1, -2, 4):
    plt.plot(r, v(r, alpha, 1, 100, 10), label=f"alpha: {alpha}")

plt.legend()
plt.xlabel('r [m]')
plt.ylabel('v [m/s]')
plt.show()
