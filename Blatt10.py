import numpy as np
import matplotlib.pyplot as plt

g = 9.81

def fall_p_z(z, m, E):
    if E < m*g*z:
        return
    return np.sqrt(2*m*(E-m*g*z))

def ozilator_p_z(z, m, E, k):
    if E < 1/2*k*z**2:
        return
    return (np.sqrt((E-1/2*k*z**2)*2*m), -np.sqrt((E-1/2*k*z**2)*2*m))

x_set_1 = np.linspace(0, 10, 10000)
m = 1
y_sets = []
E_set = np.linspace(0, 10, 3)
for E in E_set:
    y_set = []
    for x in x_set_1:
        y_set.append(fall_p_z(x, m, E))
    y_sets.append(y_set)

print("-------Plot of fall_p_z---------")
for i in range(len(y_sets)):
    plt.plot(x_set_1, y_sets[i], label=f"E: {E_set[i]}")
plt.legend()
plt.xlabel('z [m]')
plt.ylabel('p_z [kg*m/s]')
plt.savefig("Blatt10_fall_p_z.png")
plt.show()

print("-------Plot of ozilator_p_z---------")
x_set_1 = np.linspace(-10, 10, 10000)
y_sets_2 = []
x_sets_2 = []
k = 1
for E in E_set:
    y_set = []
    x_set = []
    for x in x_set_1:
        result = ozilator_p_z(x, m, E, k)
        print(result)
        if result == None:
            continue
        elif len(result) > 1:
            print("Yes")
            y_set.append(result)
            x_set.append(x)
        else:
            x_set.append(x)
            y_set.append(ozilator_p_z(x, m, E, k)[0])
    print("---------------")
    print(y_set)
    x_sets_2.append(x_set)
    y_sets_2.append(y_set)

for i in range(len(y_sets_2)):
    plt.plot(x_sets_2[i], y_sets_2[i], label=f"E: {E_set[i]}")
plt.legend()
plt.xlabel('z [m]')
plt.ylabel('p_z [kg*m/s]')
plt.savefig("Blatt10_ozilator_p_z.png")
plt.show()