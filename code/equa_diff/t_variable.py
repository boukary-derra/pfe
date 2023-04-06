import matplotlib.pyplot as plt
import numpy as np

def equa_diff_solution(input, t, tau, const):
    return const*np.exp(2*(-1/tau)*t)+input

# Générer des données pour x, y
t_data = np.linspace(1, 10, 10)

y = equa_diff_solution(input=0, t=t_data, tau=3, const=1)

# Tracer la fonction gaussienne
plt.plot(t_data, y)
plt.xlabel('Axe t: le temps')
plt.ylabel("Axe s: solution de l'equa diff")
plt.title("Variation de la solution de l'équa diff en fonction du temps t")
plt.show()
