import numpy as np
from fractions import Fraction

# la fonction Gaussian
def gaussian_function(x, y, sigma):
    G = (1/(2*np.pi*sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    return G

# La fonction pour généré la matrice 3x3 (le filtre Gaussian)
def mask_gaussian(sigma, size=3):
    mask = np.zeros((size, size))
    for x in range(size-1):
        for y in range(size-1):
            r = gaussian_function(x, y, sigma)
            #r = round(r, 6)
            mask[x, y] = r
            mask = mask/np.sum(mask)

    return mask

def gaussian_kernel(size, sigma)

    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2* sigma**2))

    kernel = kernel / (2*np.pi*sigma**2)
    return kernel
    
# Dans cette section vous créer votre mask en choisissant sigma Ex: pour sigma = 0.8 ou 1
print("Le Filtre Gaussian pour sigma = 0.8 :\n")
mask = mask_gaussian(0.7)
print(mask)

print("\n============================= SPACE   ====================================\n")

print("Le Filtre Gaussian pour sigma = 1 :\n")
mask = mask_gaussian(1)
print(mask)
