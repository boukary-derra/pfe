import matplotlib.pyplot as plt
import numpy as np

def equa_diff_solution(input, t, tau, const):
    return const*np.exp(2*(-1/tau)*t)+input

# Générer des données pour x, y
tau_data = np.linspace(1, 50, 50)

y = equa_diff_solution(input=0, t=1, tau=tau_data, const=1)

# Tracer la fonction gaussienne
plt.plot(tau_data, y)
plt.xlabel('Axe tau')
plt.ylabel("Axe s: solution de l'equa diff")
plt.title("Variation de la solution de l'équa diff en fonction de tau")
plt.show()
