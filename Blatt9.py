from scipy.integrate import quad
from scipy.optimize import root
import matplotlib.pyplot as plt
import numpy as np

# Define the integrand
#a = lambda/c
#the mass is 1

print("--------------Calculation of lambda/c-------------")

def integrand(x, a):
    return np.sqrt(1+ (np.sinh(x*np.arccosh(a))*np.cosh(a)/9.81*1*6)**2)

# Define the function to compute the difference from the target value
def difference(a, target):
    result, _ = quad(integrand, -1, 1, args=(a,))
    return result - target

# Target value for the integral
target_value = 6.0

# Initial guess for the parameter 'a'
initial_guess = 2

# Use root finding to solve the equation difference(a, target_value) = 0
result = root(difference, initial_guess, args=(target_value,))

# Print the result
print("Value of 'a'(=lambda/c) such that the integral equals the target value:", result.x[0])
print("Integral value at the optimal 'a':", quad(integrand, 0, 1, args=(result.x[0],))[0])
a = result.x[0]
print("------------------Plot the chain------------------")
print(a)
p = 1/6
g = 9.81
c = 1
y = a
def z(x):
    return (np.cosh(x*np.arccosh(a))*c-y)/(p*g)

x_s = np.linspace(-1, 1, 100)
data = []
for x in x_s:
    data.append((x, z(x)))

data = np.array(data)
plt.plot(data[:,0], data[:,1])
plt.ylabel(r"z (height)")
plt.xlabel(r"x (weight)")
plt.title("Modell of a chain")
plt.legend()
plt.savefig("Plot_chain.png")
plt.show()