from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

lambda_m = 1
def f(t, y):
    return [-np.sin(y[1]) *(1+np.cos(y[1]/(1+lambda_m))*y[0]**2)/(1-np.cos(y[1])**2/(1+lambda_m)), y[0]]

#method to calculate the results of 5.2 (the plots form last week)
def analytic_result(t, phi_0):
    t_fit = np.linspace(t[0], t[1], 100)
    #w is not relevant, because t is t tilde and t tilde = w * t
    #--> w can be replaced
    data_fit_phi = []
    for i in t_fit:
        data_fit_phi.append([phi_0 * np.cos(i)])
    return t_fit, data_fit_phi


print("-----------Test of the numerical calculation------------")
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
plt.xlabel(r"t in $\omega$ s")
plt.show()

print("----------Compare of the numerical result and the analytic one-------")
#calculate at first the numerical values
phi_0 = 1/60 *np.pi
t_span = [0, 4*np.pi]
y_0 = [0, phi_0]
t_eval = np.linspace(t_span[0], t_span[1], 100)
results_y = solve_ivp(f, t_span, y_0, t_eval=t_eval)
t_n_s = results_y.t
y_n_s = results_y.y[1]

#plot the results of the numerical calculation
plt.plot(t_n_s, y_n_s, label="numerical result")
plt.ylabel(r"$\phi$ in rad")
plt.xlabel(r"t in $\omega$ s")

t_a_s, y_a_s = analytic_result(t_span, phi_0)
print(t_a_s)
print("-----")
print(y_a_s)

plt.plot(t_a_s, y_a_s, label="analytic result")
plt.legend()
plt.savefig("Blatt6_kleinwinkelnäherung_vergleich.png")
plt.show()

print("-------------Graph with larger phi----------")
phi_0 = 1/3 *np.pi
t_span = [0, 10*np.pi]
y_0 = [0, phi_0]
t_eval = np.linspace(t_span[0], t_span[1], 150)
results_y = solve_ivp(f, t_span, y_0, t_eval=t_eval)
t_n_l = results_y.t
y_n_l = results_y.y[1]

#plot the results of the numerical calculation
plt.plot(t_n_l, y_n_l, label="numerical result")
plt.ylabel(r"$\phi$ in rad")
plt.xlabel(r"t in $\omega$ s")
plt.legend()
plt.savefig("Blatt6_phi_with_large_amplitudes_1.png")
plt.show()

print("-----------Compare the results with lambda_m value 1, goes to 0 and infinity---------")
#the results with lambda_m = 0
lambda_m = 0
phi_0 = 1/3*np.pi
t_span = [0, 10*np.pi]
y_0 = [0, phi_0]
t_eval = np.linspace(t_span[0], t_span[1], 150)
results_y = solve_ivp(f, t_span, y_0, t_eval=t_eval)
t_n_l_0 = results_y.t
y_n_l_0 = results_y.y[1]

#plot the results of the numerical calculation
plt.plot(t_n_l_0, y_n_l_0, label="numerical result")
plt.ylabel(r"$\phi$ in rad")
plt.xlabel(r"t in $\omega$ s")
plt.legend()
plt.savefig("Blatt6_phi_with_large_amplitudes_0.png")
plt.show()

#the results with lambda_m = infinity
lambda_m = np.infty
phi_0 = 1/3*np.pi
t_span = [0, 10*np.pi]
y_0 = [0, phi_0]
t_eval = np.linspace(t_span[0], t_span[1], 150)
results_y = solve_ivp(f, t_span, y_0, t_eval=t_eval)
t_n_l_inf = results_y.t
y_n_l_inf = results_y.y[1]

#plot the results of the numerical calculation
plt.plot(t_n_l_inf, y_n_l_inf, label="numerical result")
plt.ylabel(r"$\phi$ in rad")
plt.xlabel(r"t in $\omega$ s")
plt.legend()
plt.savefig("Blatt6_phi_with_large_amplitudes_inf.png")
plt.show()
