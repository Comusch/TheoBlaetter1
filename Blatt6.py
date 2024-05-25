from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

lambda_m = 1
def f(t, y):
    return [-np.sin(y[1]) *(1+np.cos(y[1]/(1+lambda_m))*y[0]**2)/(1-np.cos(y[1])**2/(1+lambda_m)), y[0]]

phi_0 = 1/5*np.pi
y_0 = [0, phi_0]
t_span= [0, 4*np.pi]
print("-------------span of time-------------")
print(t_span)

results_y = solve_ivp(f, t_span, y_0)
print("-------------numerical result-------------")
print(results_y)

print("--------get data form the results array--------")
t_array = results_y.t
y_array = results_y.y[1]
print("phi over the time data:")
print(y_array)

print("--------Plot the graph of the results--------")
plt.plot(t_array, y_array, label="numerical result")
plt.ylabel(r"$\phi$ in rad")
plt.xlabel(r"t in s")
plt.show()


