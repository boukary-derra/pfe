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

        """if (scale_percent is not None) and (isinstance(scale_percent, int)):
            input = resize_image(input, scale_percent)"""

        # Convert the image to float32 to prevent overflow
        input = input.astype(np.float32) / 255.0

        # inputs
        self.input = input
        # self.scale_percent = scale_percent

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
        self.tau_delay = 3

        # outpus
        self.gaussian_kernel = None
        self.photoreceptor_output = None
        self.lipetz_transformation_output = None
        self.low_pass_filter_output = None
        self.lmc_output = None
        self.S_ON = None
        self.S_OFF = None
        self.F_ON = None
        self.F_OFF = None
        self.output = None

        # run LMC()
        self.LMC()

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
        low_pass_image = low_pass_filter(self.photoreceptor_output, self.lipetz_transformation_tau)

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
        try:
            x = self.low_pass_filter_output
            X_lmc = low_pass_filter(x, self.lmc_tau)
            Y_lmc = x - X_lmc

            # Y_lmc = (Y_lmc * 255).astype(np.uint8)
            # Y_lmc = cv2.resize(Y_lmc, (self.width, self.height),
                                    # interpolation=cv2.INTER_AREA)

            # Convert the image back to the original range (0-255)
            # Y_lmc = (Y_lmc * 255).astype(np.uint8)

            self.lmc_output = Y_lmc

        except Exception as e:
            print("ERROR in LMC", e)

    def get_std(self):
        self.LMC()
        output = self.lmc_output
        return output

    def get_stmd(self, lmc, pre_lmc):
        # ON / OFF channels
        y_on = (lmc + abs(lmc))/2
        y_off = (lmc - abs(lmc))/2

        pre_y_on = (pre_lmc + abs(pre_lmc))/2
        pre_y_off = (pre_lmc - abs(pre_lmc))/2

        """ ====== FDSR ====== """
        if (y_on is not None) and (y_off is not None):
            try:
                s_on = y_on.copy()
                s_off = y_off.copy()


                # Loop through each pixel of the image and apply Gaussian blur
                for i, j in itertools.product(range(y_on.shape[0]), range(y_on.shape[1])):
                    # equation 10
                    if y_on[i,j] > pre_y_on[i,j]:
                        # blurred_pixel_on = low_pass_filter(y_on[i, j], self.tau_fast)
                        blurred_pixel_on = cv2.GaussianBlur(y_on[i,j], (3, 3), sigmaX=self.tau_fast, sigmaY=self.tau_fast)
                    else:
                        # blurred_pixel_on = low_pass_filter(y_on[i, j], self.tau_slow)
                        blurred_pixel_on = cv2.GaussianBlur(y_on[i,j], (3, 3), sigmaX=self.tau_slow, sigmaY=self.tau_slow)
                    # equation 11
                    if y_off[i,j] > pre_y_off[i,j]:
                        # blurred_pixel_off = low_pass_filter(y_off[i, j], self.tau_fast)
                        blurred_pixel_off = cv2.GaussianBlur(y_off[i,j], (3, 3), sigmaX=self.tau_fast, sigmaY=self.tau_fast)
                    else:
                        # blurred_pixel_off = low_pass_filter(y_off[i, j], self.tau_fast)
                        blurred_pixel_off = cv2.GaussianBlur(y_off[i,j], (3, 3), sigmaX=self.tau_slow, sigmaY=self.tau_slow)

                    blurred_pixel_on = np.mean(blurred_pixel_on)
                    blurred_pixel_off = np.mean(blurred_pixel_off)


                    s_on[i,j] = blurred_pixel_on
                    s_off[i,j] = blurred_pixel_off

                # outputs
                self.S_ON = s_on
                self.S_OFF = s_off

            except Exception as e:
                self.S_ON = None
                self.S_OFF = None
                print("ERROR in FDSR", e)
        else:
            raise ValueError("ERROR in FDSR: Y_ON, Y_OFF are empty")

        """ ====== SIGMA ====== """
        if self.S_ON is not None:
            self.F_ON = cv2.subtract(y_on, self.S_ON)
            self.F_OFF = cv2.subtract(y_off, self.S_OFF)
        else:
            raise ValueError("ERROR in SIGMA: S_ON, S_OFF are empty")

        """ ====== HW-R ======"""
        if self.F_ON is not None:
            self.F_ON = get_max(self.F_ON)
            self.F_OFF = get_max(self.F_OFF)
        else:
            raise ValueError("ERROR in HW-R: F_ON, F_OFF are empty")

        """ ====== LI: pass ====== """

        """ ====== DELAY ====== """
        if self.F_OFF is not None:
            self.F_OFF = low_pass_filter(self.F_OFF, self.tau_delay)
        else:
            raise ValueError("ERROR in DELAY: F_OFF is empty")

        """ ====== FINAL OUTPUT ====== """
        if (self.F_ON is not None) and (self.F_OFF is not None):
            self.output = cv2.multiply(self.F_ON, self.F_OFF)
        else:
            raise ValueError("ERROR in FINAL OUTPUT: F_ON or F_OFF are empty")

        return self.output

    """def get_delta(self, frame, pre_frame):
        " to remove static targets "
        delta = cv2.subtract(frame, pre_frame)
        return delta"""

    def convert_for_display(self, input):
        # Convert the image back to the original range (0-255)
        input = (input * 255).astype(np.uint8)
        #input = cv2.resize(input, (self.width, self.height), interpolation=cv2.INTER_AREA)
        input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
        return input

def low_pass_filter(image, tau):
    # Apply a simple low-pass filter to the image
    kernel_size = int(2 * np.ceil(2 * tau) + 1)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=tau, sigmaY=tau)
    return blurred_image


"""def resize_image(image, scale_percent):

    # Calculate the new dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)

    # Resize the image
    image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return image"""

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
