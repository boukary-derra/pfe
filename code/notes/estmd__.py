#======= Boukary DERRA ==========

# Modules
import numpy as np
import scipy
from scipy import ndimage


class ESTMD:
    def __init__(self, frame):
        # Convert input frame to Gray Syle
        # kernel
        self.gauss_filter_size = 3
        self.gauss_filter_sigma = 0.8
        self.input = frame


    """ =============================== TOOLS =============================="""
    def get_gaussian_mask(self):
        try:
            size, sigma = self.gauss_filter_size, self.gauss_filter_sigma
            # création d'un ax
            ax = np.arange(-size // 2 + 1, size // 2 + 1)
            xx, yy = np.meshgrid(ax, ax)

            kernel = np.exp(-(xx**2 + yy**2) / (2* sigma**2))

            kernel = kernel / (2*np.pi*sigma**2)

            self.gaussian_mask = kernel
        except Exception as e:
            self.gaussian_mask = None
            print("Error in Guassian Kernel :", e)



    """ ================= Retina Layer ========================="""
    def photoreceptor(self):
        """ Retina Layer """
        self.get_gaussian_mask()
        if self.gaussian_mask is not None:
            # self.photoreceptors_output = scipy.signal.convolve2d(self.input, self.gauss_filter, mode='same')
            self.photoreceptors_output = ndimage.convolve(self.input, self.gaussian_mask)
            print(self.photoreceptors_output)
        else:
            self.photoreceptors_output = None
            print("Error in photoreceptor :", e)
