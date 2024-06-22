from asyncio import sleep

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import _thread

# Define the differential equation as a function
def f(t, y, lambda_m):
    epsilon = 1e-12  # Small value to avoid division by zero
    cos_term = np.cos(y[1])
    denominator = 1 - cos_term ** 2 / (1 + lambda_m)

    # Adjust the denominator to avoid division by zero
    if abs(denominator) < epsilon:
        denominator = epsilon if denominator > 0 else -epsilon

    dydt = -np.sin(y[1]) * (1 + np.cos(y[1] / (1 + lambda_m)) * y[0] ** 2) / denominator
    return [dydt, y[0]]


# Method to calculate the analytical results (assuming a specific form of the analytical solution)
def analytic_result(t, phi_0):
    t_fit = np.linspace(t[0], t[1], 100)
    data_fit_phi = [phi_0 * np.cos(1.4*i) for i in t_fit]
    return t_fit, data_fit_phi


# Function to solve the ODE with given parameters and plot the results
def solve_and_plot(t_span, y_0, lambda_m, t_eval, title, filename, save=True):
    results_y = solve_ivp(f, t_span, y_0, args=(lambda_m,), t_eval=t_eval)
    t_n = results_y.t
    y_n = results_y.y[1]

    # Plot the results of the numerical calculation#
    if save:
        plt.plot(t_n, y_n, label=f'numerical result (lambda_m={lambda_m})')
        plt.ylabel(r"$\phi$ in rad")
        plt.xlabel(r"t in $\omega$ s")
        plt.title(title)
        plt.legend()
    if save:
        plt.savefig(filename)
        plt.show()
    return results_y


# Initial conditions and time span
phi_0 = 1 / 5 * np.pi
y_0 = [0, phi_0]
t_span = [0, 4 * np.pi]

# Numerical result for default parameters
print("-----------Test of the numerical calculation------------")
results_y = solve_ivp(f, t_span, y_0, args=(1,))
print("-------------numerical result-------------")
print(results_y)

# Extract data from results
t_array = results_y.t
y_array = results_y.y[1]
print("phi over the time data:")
print(y_array)

# Plot the numerical result
plt.plot(t_array, y_array, label="numerical result")
plt.ylabel(r"$\phi$ in rad")
plt.xlabel(r"t in $\omega$ s")
plt.show()

# Comparison of the numerical result and the analytical one
print("----------Compare the numerical result and the analytic one-------")
phi_0 = 1 / 60 * np.pi
y_0 = [0, phi_0]
t_eval = np.linspace(t_span[0], t_span[1], 100)
results_y = solve_ivp(f, t_span, y_0, args=(1,), t_eval=t_eval)
t_n_s = results_y.t
y_n_s = results_y.y[1]

# Plot the numerical result
plt.plot(t_n_s, y_n_s, label="numerical result")

# Calculate the analytical result
t_a_s, y_a_s = analytic_result(t_span, phi_0)
plt.plot(t_a_s, y_a_s, label="analytic result")
plt.ylabel(r"$\phi$ in rad")
plt.xlabel(r"t in $\omega$ s")
plt.legend()
plt.savefig("Blatt6_kleinwinkelnäherung_vergleich.png")
plt.show()

# Graph with larger phi
print("-------------Graph with larger phi----------")
phi_0 = 1 / 3 * np.pi
y_0 = [0, phi_0]
t_span = [0, 10 * np.pi]
t_eval = np.linspace(t_span[0], t_span[1], 150)
solve_and_plot(t_span, y_0, 1, t_eval, "Larger Amplitudes (lambda_m = 1)", "Blatt6_phi_with_large_amplitudes_1.png")

# Compare the results with different lambda_m values
print("-----------Compare the results with lambda_m values 1, 0, and infinity---------")

# lambda_m = 0
solve_and_plot(t_span, y_0, 0, t_eval, "Larger Amplitudes (lambda_m = 0)", "Blatt6_phi_with_large_amplitudes_0.png")

# lambda_m = infinity
solve_and_plot(t_span, y_0, np.inf, t_eval, "Larger Amplitudes (lambda_m = inf)",
               "Blatt6_phi_with_large_amplitudes_inf.png")

print("-----------rollover with v_0 ------------")
for v_0 in np.linspace(0, 3, 100):
    y_0 = [v_0, 1/3*np.pi]
    t_span = [0, 200 * np.pi]
    t_eval = np.linspace(t_span[0], t_span[1], 150)
    results_y = solve_and_plot(t_span, y_0, 1, t_eval, f"Larger Amplitudes (lambda_m = 1 and v_0 = {y_0[0]})", "Blatt6_phi_with_large_amplitudes_1_v0.png", False)
    for data in results_y.y[1]:
        if data > np.pi or data < -1*np.pi:
            print(f"With the beginning speed of v_0 {v_0}, there is a rollover.")
            break

