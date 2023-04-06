import matplotlib.pyplot as plt
import numpy as np

# Définir la fonction gaussienne 2D
def gauss(x, y, a, mu, sigma):
    return a*np.exp(-(x - mu)**2 / (2 * sigma**2)) * np.exp(-(y - mu)**2 / (2 * sigma**2))

# Générer des données pour x, y et z
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = gauss(X, Y, 1, 0, 1)

# Tracer la fonction gaussienne 2D
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, cmap="plasma")
ax.set_title('Fonction gaussienne bidimensionnelle')
ax.set_xlabel("Axe X")
ax.set_ylabel("Axe Y")
ax.set_zlabel("Axe Z")
plt.show()
