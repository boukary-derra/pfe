import matplotlib.pyplot as plt
import numpy as np

def equa_diff_solution(input, t, tau, const):
    return const*np.exp(2*(-1/tau)*t)+input

# Générer des données pour x, y et z
tau_data = np.linspace(1, 501, 1000)
const_data = np.linspace(0, 500, 1000)
X, Y = np.meshgrid(tau_data, const_data)
Z = equa_diff_solution(input=0, t=1, tau=X,const=Y)

# Tracer la fonction gaussienne 2D
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, cmap="plasma")
ax.set_title("Variation de la solution de l'équa diff en fonction de tau et const")
ax.set_xlabel("Axe X: TAU")
ax.set_ylabel("Axe T: CONST")
ax.set_zlabel("Axe Z: Solution de l'equa diff")
plt.show()
