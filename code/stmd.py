#======= Boukary DERRA ==========
# ===== PFE: 01-03-2023 ===========

# Modules
import numpy as np
import itertools
import cv2
import matplotlib.pyplot as plt

# class STMD
class STMD:
    def __init__(self, input=None, scale_percent=None):

        # checks
        if (input is None) or (not isinstance(input, np.ndarray)):
            raise ValueError("Input must be an image")

        if len(input.shape) > 2:
            input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

        self.height, self.width = input.shape[:2]

        if (scale_percent is not None) and (isinstance(scale_percent, int)):
            input = resize_image(input, scale_percent)

        # Convert the image to float32 to prevent overflow
        input = input.astype(np.float32) / 255.0

        # inputs
        self.input = input
        self.scale_percent = scale_percent

        # constants
        self.gauss_filter_size = 3
        self.gauss_filter_sigma = 0.8
        self.u = 0.7
        self.w = np.array([1/16, 1/8, 1/16, 1/8, 1/4,
                                1/8, 1/16, 1/8, 1/16]).reshape(3, 3)
        self.lipetz_transformation_tau = 3
        self.low_pass_filter_tau = 3
        self.lmc_tau = 3
        self.tau_fast = 5
        self.tau_slow = 15

        # outpus
        self.gaussian_kernel = None
        self.photoreceptor_output = None
        self.lipetz_transformation_output = None
        self.low_pass_filter_output = None
        self.lmc_output = None

        # run


    def get_gaussian_kernel(self):
        try:
            size, sigma = self.gauss_filter_size, self.gauss_filter_sigma

            ax = np.arange(-size // 2 + 1, size // 2 + 1)
            xx, yy = np.meshgrid(ax, ax)

            kernel = np.exp(-(xx**2 + yy**2) / (2* sigma**2))

            kernel = kernel / (2*np.pi*sigma**2)

            self.gaussian_kernel = kernel
        except Exception as e:
            self.gaussian_kernel = self.w
            print("Error in Guassian Kernel :", e)



    """ ================= Retina Layer ========================="""
    def photoreceptor(self):
        """ Blur effect """
        # input frame size (m: with, n: height)
        self.get_gaussian_kernel()
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
                            l += self.gaussian_kernel[u, v] * self.input[i+u, j+v]
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
        low_pass_image = low_pass_filter(self.photoreceptor_output,
                                            self.lipetz_transformation_tau)

        # Apply the modified Lipetz transformation
        numerator = np.power(self.photoreceptor_output, self.u)

        denominator = np.power(self.photoreceptor_output, self.u) + np.power(low_pass_image, self.u)

        transformed_image = np.divide(numerator, denominator)

        self.lipetz_transformation_output = transformed_image


    """ ======================= Lamina Layer ==============================="""
    def low_pass_filter(self):
        """ Slight delay """
        self.lipetz_transformation()
        self.low_pass_filter_output = low_pass_filter(self.lipetz_transformation_output, self.low_pass_filter_tau)


    def LMC(self):
        """ (LMCs) Remove redundant information; Maximize information
                                transmission                            """
        self.low_pass_filter()

        x = self.low_pass_filter_output
        X_lmc = low_pass_filter(x, self.lmc_tau)
        Y_lmc = x - X_lmc

        # Y_lmc = (Y_lmc * 255).astype(np.uint8)
        # Y_lmc = cv2.resize(Y_lmc, (self.width, self.height),
                                # interpolation=cv2.INTER_AREA)

        # Convert the image back to the original range (0-255)
        Y_lmc = (Y_lmc * 255).astype(np.uint8)
        
        self.lmc_output = Y_lmc

        # ON / OFF channels
        # self.lmc_output_ON = (Y_lmc + abs(Y_lmc))/2
        # self.lmc_output_OFF = (Y_lmc - abs(Y_lmc))/2

    def FDSR(self, pre_lmc):
        pass
        """if self.lmc_output.shape == pre_lmc.shape:
            # Calculate the difference between the two images
            self.delta = cv2.subtract(self.lmc_output, pre_lmc)

            # Fast Depolarization
            fast_depolarization_on = cv2.GaussianBlur(delta_on, (0, 0), self.tau_fast)
            fast_depolarization_off = cv2.GaussianBlur(delta_off, (0, 0), self.tau_fast)

            # Slow Repolarization
            slow_repolarization_on = cv2.GaussianBlur(fast_depolarization_on, (0, 0), self.tau_slow)
            slow_repolarization_off = cv2.GaussianBlur(fast_depolarization_off, (0, 0), self.tau_slow)

            # Calculate S by adding the slow repolarization output to the image at time t
            S_on = cv2.add(self.lmc_output_ON, slow_repolarization_on)
            S_off = cv2.add(self.lmc_output_OFF, slow_repolarization_off)

            # outputs
            self.FDSR_ON_ouptut = S_on
            self.FDSR_OFF_ouptut = S_off
        else:
            raise ValueError ("frame and pre_frame must be the same frame")"""


    def get_std(self):
        self.LMC()
        output = self.lmc_output
        return output

    def get_delta(self, frame, pre_frame):
        " to remove static targets "
        delta = cv2.subtract(frame, pre_frame)
        return delta

    def get_fdsr(self, frame, delta):
        # Fast Depolarization
        fast_depolarization = cv2.GaussianBlur(delta, (0, 0), self.tau_fast)
        # Slow Repolarization
        slow_repolarization = cv2.GaussianBlur(fast_depolarization, (0, 0), self.tau_slow)
        # Calculate S by adding the slow repolarization output to the image at time t
        S = cv2.add(frame, slow_repolarization)
        return S

    def convert_for_display(self, input):
        input = cv2.resize(input, (self.width, self.height), interpolation=cv2.INTER_AREA)
        input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        return input

def low_pass_filter(image, tau):
    # Apply a simple low-pass filter to the image
    kernel_size = int(2 * np.ceil(2 * tau) + 1)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=tau, sigmaY=tau)
    return blurred_image


def resize_image(image, scale_percent):

    # Calculate the new dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)

    # Resize the image
    image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return image
