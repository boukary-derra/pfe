#======= Boukary DERRA ==========

# Modules
from std import STD
import numpy as np
import itertools
import cv2
import matplotlib.pyplot as plt


class STMD(STD):
    def __init__(self, input=None, pre_input=None, scale_percent=None):

        # checks

        input = STD(input, scale_percent).LMC().lmc_output
        pre_input = STD(pre_input, scale_percent).lmc_output

        # inputs
        self.input = input
        pre_input.input = pre_input
        self.scale_percent = scale_percent

        # constants
        self.gauss_filter_size = 3
        self.gauss_filter_sigma = 0.8
        self.u = 0.7
        self.tau_1 = 3
        self.tau_2 = 3
        self.tau_3 = 3
        self.tau_fast = 5
        self.tau_slow = 15
        self.tau_5 = 3
        self.const_1 = 5
        self.const_2 = 12
        self.const_3 = 8
        self.const_4 = 12
        self.const_5 = 8

        # outpus
        self.gaussian_mask = None
        self.photoreceptor_output = None
        self.lipetz_transformation_output = None
        self.low_pass_filter_output = None
        self.lmc_output = None
        self.lmc_output_ON = None
        self.lmc_output_OFF = None
        self.FDSR_ON_ouptut = None
        self.FDSR_OFF_ouptut = None
        self.HWR_ON_output = None
        self.HWR_OFF_output = None
        self.LI_ON_output = None
        self.LI_OFF_output = None
        self.LI_OFF_delay_output = None
        self.output = None


    """ =============================== TOOLS =============================="""

    """def get_equa_diff_solution(self, input, t, tau, const):
        return const*np.exp((-1/tau)*t)+input"""


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
            print("Error in Guassian Kernel :", e)
            self.gaussian_mask = np.array([1/16, 1/8, 1/16, 1/8, 1/4, 1/8, 1/16, 1/8, 1/16]).reshape(3, 3)


    """ ================= Retina Layer ========================="""
    def photoreceptor(self):
        """ Blur effect"""
        # input frame size (m: with, n: height)
        self.get_gaussian_mask()
        (m, n) = self.input.shape

        try:
            # output initialization (null matrix with the same size as the input)
            buffer_frame = np.zeros((m, n))

            # iterate through all the elements of the input frame
            for (i, j) in itertools.product(range(m), range(n)):
                l = 0

                # equation [1]
                # sum of v from -1 to 1
                for v in [-1, 0, 1]:
                    # sum of u from -1 to 1
                    for u in [-1, 0, 1]:
                        try:
                            l += self.gaussian_mask[u, v] * self.input[i+u, j+v]
                        except: pass

                 # fill output frame element by element
                buffer_frame[i, j] = l

            self.photoreceptor_output  = buffer_frame

        except Exception as e:
            print("Error in Retina layer / photoreceptor :", e)

    def lipetz_transformation(self):
        """ transform the input luminance to membrane potential """
        self.photoreceptor()

        # Calculate the low-pass filtered version of the image
        low_pass_image = low_pass_filter(self.photoreceptor_output, self.tau_1)

        # Apply the modified Lipetz transformation
        numerator = np.power(self.photoreceptor_output, self.u)
        denominator = np.power(self.photoreceptor_output, self.u) + np.power(low_pass_image, self.u)
        transformed_image = np.divide(numerator, denominator)


        self.lipetz_transformation_output = transformed_image


    """ ======================= Lamina Layer ==============================="""
    def low_pass_filter(self):
        """ Slight delay """
        self.lipetz_transformation()

        self.low_pass_filter_output = low_pass_filter(self.lipetz_transformation_output, self.tau_1)


    def LMC(self):
        """ (LMCs) Remove redundant information; Maximize information transmission """
        self.low_pass_filter()

        x = self.low_pass_filter_output
        X_lmc = low_pass_filter(x, self.tau_1)
        Y_lmc = x - X_lmc


        self.lmc_output = Y_lmc
        self.lmc_output_ON = (Y_lmc + abs(Y_lmc))/2
        self.lmc_output_OFF = (Y_lmc - abs(Y_lmc))/2


    """ ======================= Medulla Layer =============================="""
    def FDSR(self):
        """ The FDSR mechanism is able to suppress rapidly changed texture
        information and enhance noval contrast change """

        pre_stmd = ESTMD(input=self.pre_input)
        pre_stmd.LMC()
        self.LMC()

        # Calculate the difference between the two images
        delta_on = cv2.subtract(self.lmc_output_ON, pre_stmd.lmc_output_ON)
        delta_off = cv2.subtract(self.lmc_output_OFF, pre_stmd.lmc_output_OFF)


        # Fast Depolarization
        fast_depolarization_on = cv2.GaussianBlur(delta_on, (0, 0), self.tau_fast)
        fast_depolarization_off = cv2.GaussianBlur(delta_off, (0, 0), self.tau_fast)

        # self.testtt = delta_off

        # Slow Repolarization
        slow_repolarization_on = cv2.GaussianBlur(fast_depolarization_on, (0, 0), self.tau_slow)
        slow_repolarization_off = cv2.GaussianBlur(fast_depolarization_off, (0, 0), self.tau_slow)

        # Calculate S by adding the slow repolarization output to the image at time t
        S_on = cv2.add(self.lmc_output_ON, slow_repolarization_on)
        S_off = cv2.add(self.lmc_output_OFF, slow_repolarization_off)

        # outputs
        self.FDSR_ON_ouptut = S_on
        self.FDSR_OFF_ouptut = S_off



    def sigma_et_HWR(self):
        self.FDSR()
        zero_array = np.zeros_like(self.input)

        F_on = self.lmc_output_ON - self.FDSR_ON_ouptut
        self.HWR_ON_output = get_max(-F_on)
        F_off = self.lmc_output_OFF - self.FDSR_OFF_ouptut
        self.HWR_OFF_output = get_max(F_off)


    """ ======================= Lobula Layer ============================ """
    def LI(self):
        self.sigma_et_HWR()

        self.LI_ON_output = self.HWR_ON_output
        self.LI_OFF_output = self.HWR_OFF_output


    def delay(self):
        """ Slight delay on the OFF channel """
        self.LI()

        self.LI_OFF_delay_output = low_pass_filter(self.LI_OFF_output, self.tau_5)

    def final_output(self):
        """ Exhibits correlation between ON and OFF channels """
        self.delay()

        self.output = cv2.multiply(self.LI_ON_output, self.LI_OFF_delay_output)

def low_pass_filter(image, tau):
    # Apply a simple low-pass filter to the image
    kernel_size = int(2 * np.ceil(2 * tau) + 1)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=tau, sigmaY=tau)
    return blurred_image

def get_max(img):
    # Compare each pixel value to 0 and get a mask with the result
    mask = cv2.compare(img, 0, cv2.CMP_GT)

    # Create an array with the same shape as the grayscale image and fill it with zeros
    zero_array = np.zeros_like(img)

    # Compute the maximum between the grayscale image and the zero array for each pixel
    max_image = cv2.max(img, zero_array)

    # Set the values in max_image to 0 where the original pixel values were negative
    max_image = cv2.bitwise_and(max_image, max_image, mask=mask)

    return max_image


def resize_image(image, scale_percent):

    # Calculate the new dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)

    # Resize the image
    image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return image

def width_to_n(image, n):
    w = n*(100/image.shape[1])
    return resize_image(image, w)

def n_to_width(image, w):
    n = w*(100/image.shape[1])
    return resize_image(image, n)
