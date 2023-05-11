#======= Boukary DERRA (PFE) ==========
# ===== created: 01-02-2023 ===========
# ===== last updated: 09-05-2023 ======

# Modules
import numpy as np
import itertools
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# class STMD
class STMD:
    def __init__(self, frame, last_frame=None):

        # checks
        if (frame is None) or (not isinstance(frame, np.ndarray)):
            raise ValueError("frame must be an image")

        # preprocessing
        frame = self.convert_for_process(frame)

        # inputs
        self.frame = frame
        self.last_frame = last_frame

        # constants
        self.height, self.width = frame.shape[:2]
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

        # ======> LI constants======
        self.k1, self.k2 = 2, 2
        self.block_size = 1
        self.search_range = 8
        self.p, self.q = 8, 8 # line height and column width of the kernel H
        self.a = 1

    def get_gaussian_kernel(self):
        pass


    # ====================> RETINA LAYER <====================
    def get_photoreceptor(self):
        """ Blur effect """
        pass

    def get_lipetz_transformation(self):
        """ transform the input luminance to membrane potential """
        pass


    # ====================> Lamina LAYER <====================
    def get_low_pass_filter(self):
        """ Slight delay """
        pass

    def get_lmc(self):
        """ (LMCs) Remove redundant information; Maximize information
                                transmission                            """
       pass


    # ====================> Medulla LAYER <====================
    def get_on_off_channels(self):
    """ Separate LMC to ON/OFF channels for next stage of the processing """
        pass

    def get_fdsr(self):
    """ The FDSR mechanism is able to suppress rapidly changed texture
        information and enhance noval contrast change """
        pass

    def get_sigma(self):
    """ Subtract the filtered signal (s_on, s_off) to the original one
			(y_on, y_off) """
        pass

    def get_hwr(self):
    """ Then we replace all the pixels that are negative with 0. """
        pass


    # ====================> Lobula LAYER <====================
    def get_li(self):
    """ LI plays a significant role in differentiating target motion
     from background motion.
	The new LIM that considers velocity and motion direction """
        pass

    def get_delay(self):
    """ Slight delay on the OFF channel """
        pass

    def get_final_output(self):
    """ Exhibits correlation between ON and OFF channels """
        pass

    def convert_for_process(self, frame):
    """ Frame preprocessing """
        pass

    def convert_for_display(self, frame):
    """ Prepare frame for display """
        Pass


# ====================> Functions <====================
def low_pass_filter(image, tau):
    """ Apply a simple low-pass filter to an image """
    pass

def get_max(img):
    """ for HW-R """
    Pass


# ======> Fonctions for LI ======
def calculate_sad(block1, block2):
    """ Sum of absolute differences between two block
            or (the matching criteria) -> equayion [6]             """
    sad =  np.sum(np.abs(block1 - block2))
    return sad

def find_optimal_motion_vector(current_block, last_frame, i, j, search_range):
    """ To calculate the translation vector (namely the motion vector)
            that minimizes the motion criteria -> equation [17]        """
    pass

def block_based_motion_estimation(current_frame, last_frame, block_size, search_range):
    """ This code divides the current frame into blocks of a given size and
        finds the optimal motion vector for each block by searching within
        the last frame. for our case, blok_size = 1.
        So it will find the optical motion for each pixel """
    pass

def get_motion_components(motion_vectors):
    """ To get the motion vector matrix U, V """
    pass

def create_convolution_kernel(p, q, a):
    """ Generate kernel H """
    pass

def convolve_uv_with_h(U, V, H):
    """ Convlolve U and V by H in order to get """
    pass

def calculate_w(U_c, V_c):
    """ Calulate W """
    pass

def lateral_inhibition(F_ON, F_OFF, w, k1, k2):
    """ hen, we implement our new lateral inhibition mechanism
        by multiplying F_ON and F_OFF by w, respectively  """
    pass
