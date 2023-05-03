import scipy
import cv2
import numpy as np


class ESTMD:
    def __init__(self, frame):
        self.gauss_filter_size = 3
        self.gauss_filter_sigma = 1

        self.gauss_filter = None
        #self.input = None
        self.photoreceptors_output = None
        self.cell_photoreceptors_output = None

        frame = cv2.imread('media/image2.jpg')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.input = frame


    """ generate gaussian filter """
    def get_gauss_filter(self):
        size, sigma = self.gauss_filter_size, self.gauss_filter_sigma
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / np.sum(kernel)

        self.gauss_filter = kernel


    """ Retina Layer """
    def retina(self):
        # perform GaussFilter to input
        self.photoreceptors_output = scipy.signal.convolve2d(self.input, self.gauss_filter, mode='same')
        # record the photoreceptors output
        if self.cell_photoreceptors_output:
            self.cell_photoreceptors_output.pop(0)
        self.cell_photoreceptors_output.append(self.photoreceptors_output)

    def lamina(self):
        pass


class Gamma_Filter:
    def __init__(self, Order1=2, Tau1=3, Len1=None, Order2=6, Tau2=9, Len2=None):
        self.Order1 = Order1
        self.Tau1 = Tau1
        self.Len1 = Len1
        self.Order2 = Order2
        self.Tau2 = Tau2
        self.Len2 = Len2

        self.Gammakernel_1 = None
        self.Gammakernel_2 = None
        self.GammaFun1_Output = None
        self.GammaFun2_Output = None

        self.Init_Gamma_kernel()

    def Init_Gamma_kernel(self):
        if self.Len1 is None:
            self.Len1 = 3 * np.ceil(self.Tau1)

        if self.Len2 is None:
            self.Len2 = 3 * np.ceil(self.Tau2)

        self.Gammakernel_1 = ToolFun().generalize_gammakernel(self.Order1, self.Tau1, self.Len1)
        self.Gammakernel_2 = ToolFun().generalize_gammakernel(self.Order2, self.Tau2, self.Len2)

    def go(self, Input):
        self.GammaFun1_Output = cell_conv_n_1(Input, self.Gammakernel_1)
        self.GammaFun2_Output = cell_conv_n_1(Input, self.Gammakernel_2)

        Filter_Output = self.GammaFun1_Output - self.GammaFun2_Output
        return Filter_Output

# You will need to provide the implementations of `generalize_gammakernel` and `cell_conv_n_1` functions
# which are being called in the `Init_Gamma_kernel` and `go` methods of the `Gamma_Filter` class.


""" ============================ """
class ToolFun:
    @staticmethod
    def generalize_gammakernel(order, tau, wide):
        if wide <= 1:
            wide = 2
        gamma = np.zeros(wide)  # initialization
        for k in range(wide):
            t = k
            gamma[k] = (order * t / tau) ** order * np.exp(-order * t / tau) / \
                       (np.math.factorial(order - 1) * tau)
        gamma = gamma / np.sum(gamma)  # normalization
        return gamma

    @staticmethod
    def generalize_fractional_derivative_kernel(alpha, wide):
        if wide < 2:
            wide = 2
        kernel_k = np.zeros(wide - 1)  # initialization
        kernel_f = np.zeros(wide)
        if alpha == 1:
            kernel_k[0] = 1
        elif 0 < alpha < 1:
            for k in range(wide - 1):
                t = k
                kernel_k[k] = np.exp(-alpha * t / (1 - alpha)) / (1 - alpha)
            kernel_k = kernel_k / np.sum(kernel_k)  # normalization
        else:
            raise ValueError("Alpha must be in the range (0, 1]")

        # Divide the difference of kernel_K
        kernel_f[0] = kernel_k[0]
        kernel_f[1:-1] = kernel_k[1:] - kernel_k[:-1]
        kernel_f[-1] = -kernel_k[-1]
        return kernel_f

    @staticmethod
    def generalize_lateral_inhibition_kernel_w2(kernel_size=15, sigma1=1.5, sigma2=3, e=1.0, rho=0, a=1, b=3):
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        cen_x = kernel_size // 2
        cen_y = kernel_size // 2
        x, y = np.meshgrid(np.arange(1, kernel_size + 1), np.arange(1, kernel_size + 1))
        shift_x = x - cen_x
        shift_y = y - cen_y
        gauss1 = (1 / (2 * np.pi * sigma1 ** 2)) * np.exp(-(shift_x * shift_x + shift_y * shift_y) / (2 * sigma1 ** 2))
        gauss2 = (1 / (2 * np.pi * sigma2 ** 2)) * np.exp(-(shift_x * shift_x + shift_y * shift_y) / (2 * sigma2 ** 2))
        dog_filter = gauss1 - e * gauss2 - rho
        positive_component = np.maximum(dog_filter, 0)
        negative_component = np.maximum(-dog_filter, 0)
        inhibition_kernel_w2 = a * positive_component - b * negative_component
        return inhibition_kernel_w2

    @staticmethod
    def generalize_dstmd_directional_inhibition_kernel(dstmd_directions=8, sigma1=1.5, sigma2=3.0):
        kernel_size = dstmd_directions
        zero_point_dog_x1 = -np.sqrt((np.log(sigma2 / sigma1) * 2 * sigma1 ** 2 * sigma2 ** 2) /
                                      (sigma2 ** 2 - sigma1 ** 2))
        zero_point_dog_x2 = -zero_point_dog_x1
        min_point_dog_x1 = -np.sqrt((3 * np.log))


""" ============================ """
def cell_conv_n_1(input_cell, kernel, head_pointer=None):
    # Input: a list where each element has the same dimension
    # Kernel: a vector
    # head_pointer: head pointer of Input
    k1 = len(input_cell)
    if head_pointer is None:
        # Set head pointer as the remaining storage matrix of the other nodes when calculating
        head_pointer = k1

    kernel = np.squeeze(kernel)
    if not np.ndim(kernel) == 1:
        raise ValueError("The Kernel must be a vector!")

    k2 = len(kernel)
    length = min(k1, k2)

    if input_cell[head_pointer-1] is None:
        return None
    else:
        output = np.zeros(np.shape(input_cell[head_pointer-1]))

    for t in range(length):
        j = (head_pointer - t) % k1
        if abs(kernel[t]) > 1e-16 and input_cell[j] is not None:
            # The value range of mod is at least 0,
            # but the index of MATLAB array starts from 1, so add 1 to make it consistent
            output += input_cell[j] * kernel[t]

    return output
