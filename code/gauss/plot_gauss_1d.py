import matplotlib.pyplot as plt
import numpy as np

# Définir la fonction gaussienne
def gauss(x, a, mu, sigma):
    return a*np.exp(-(x - mu)**2 / (2 * sigma**2))

# Générer des données pour x, y
x = np.linspace(-5, 5, 100)

y = gauss(x, 1, 0, 1)
# Tracer la fonction gaussienne
print(x)
print(y)
plt.plot(x, y)
plt.xlabel('Axe x')
plt.ylabel('Axe y')
plt.title('Fonction Gaussienne')
plt.show()
