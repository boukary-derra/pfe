import matplotlib.pyplot as plt
import numpy as np

def equa_diff_solution(input, t, tau, const):
    return const*np.exp(2*(-1/tau)*t)+input

# Générer des données pour x, y
const_data = np.linspace(-49, 50, 100)
print(const_data)

y = equa_diff_solution(input=0, t=1, tau=3, const=const_data)

# Tracer la fonction gaussienne
plt.plot(const_data, y)
plt.xlabel('Axe CONST')
plt.ylabel("Axe s: solution de l'equa diff")
plt.title("Variation de la solution de l'équa diff en fonction de CONST")
plt.show()
