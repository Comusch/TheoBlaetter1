import matplotlib.pyplot as plt
import numpy as np

g = 9.81

def w(l, m1, m2):
    return np.sqrt(g/l*(m1+m2)/m1)


phi_0 = np.pi*1/12
data_fit_q = []
data_fit_phi = []
for i in range(1, 4):
    data_fit_q_neu = []
    data_fit_phi_neu = []
    m1 = 1
    m2 = i
    l = 1
    w_neu = w(l, m1, m2)
    for i in np.linspace(0, 2, 100):
        data_fit_q_neu.append([i, -phi_0*l*m2/(m1+m2)*np.cos(w_neu*i)+ phi_0*l*m2/(m1+m2)])
        data_fit_phi_neu.append([i, phi_0*np.cos(w_neu*i)])
    data_fit_q.append(data_fit_q_neu)
    data_fit_phi.append(data_fit_phi_neu)

z = 1
for i in data_fit_q:
    array_x = np.array(i)[:,0]
    array_y = np.array(i)[:,1]
    #es gibt nur 4 graphen, weil es nur 4 m2 gibt
    plt.plot(array_x, array_y, label=f"m2: {z}")
    plt.legend()
    plt.xlabel('t [s]')
    plt.ylabel('q [m]')
    z +=1
plt.savefig("Blatt5_q.png")
plt.show()

z= 1
for i in data_fit_phi:
    array_x = np.array(i)[:,0]
    array_y = np.array(i)[:,1]
    #es gibt nur 4 graphen, weil es nur 4 m2 gibt
    plt.plot(array_x, array_y, label=f"m2: {z}")
    plt.legend()
    plt.xlabel('t [s]')
    plt.ylabel('phi [rad]')
    z +=1
plt.savefig("Blatt5_phi.png")
plt.show()
