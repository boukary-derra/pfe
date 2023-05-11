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
        self.height, self.width = frame.shape[:2]
        # inputs
        self.frame = frame
        self.last_frame = last_frame

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

        # ======> LI constants ======
        self.k1, self.k2 = 2, 2
        self.block_size = 1
        self.search_range = 8
        self.p, self.q = 8, 8  # line height and column width of the kernel H
        self.a = 1


    def get_gaussian_kernel(self):
        try:
            # to delemite the axes
            size, sigma = self.gauss_filter_size, self.gauss_filter_sigma

            ax = np.arange(-size // 2 + 1, size // 2 + 1)
            xx, yy = np.meshgrid(ax, ax)

            kernel = np.exp(-(xx**2 + yy**2) / (2* sigma**2))

            kernel = kernel / (2*np.pi*sigma**2)

            gaussian_kernel = kernel
        except Exception as e:
            gaussian_kernel = self.w
            print("Error in Guassian Kernel :", e)
        return gaussian_kernel


    """ ================= Retina Layer ========================="""
    def get_photoreceptor(self):
        """ Blur effect """
        # input frame size (m: with, n: height)
        gaussian_kernel = self.get_gaussian_kernel()
        (m, n) = self.frame.shape

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
                            l += gaussian_kernel[u, v] * self.frame[i+u, j+v]
                        except: pass
                 # fill output frame element by element
                buffer_frame[i, j] = l
            l = buffer_frame

        except Exception as e:
            photoreceptor = None
            print("Error in Retina layer / photoreceptor :", e)
        return l

    def get_lipetz_transformation(self):
        """ transform the input luminance to membrane potential """
        l = self.get_photoreceptor()

        # Calculate the low-pass filtered version of the image
        lc = low_pass_filter(l, self.lipetz_transformation_tau)

        # Apply the modified Lipetz transformation
        numerator = np.power(l, self.u)

        denominator = np.power(l, self.u) + np.power(lc, self.u)

        # transformed image
        p = np.divide(numerator, denominator)

        return p


    """ ======================= Lamina Layer ==============================="""
    def get_low_pass_filter(self):
        """ Slight delay """
        p = self.get_lipetz_transformation()
        x = low_pass_filter(p, self.low_pass_filter_tau)
        return x

    def get_lmc(self):
        """ (LMCs) Remove redundant information; Maximize information
                                transmission                            """
        try:
            x = self.get_low_pass_filter()
            X_lmc = low_pass_filter(x, self.lmc_tau)
            y_lmc = x - X_lmc


        except Exception as e:
            y_lmc = None
            print("ERROR in LMC", e)

        return y_lmc


    # ====================> Medulla LAYER <====================
    def get_on_off_channels(self):
        """ Separate LMC to ON/OFF channels for next stage of the processing """
        y_lmc = self.get_lmc()
        if y_lmc is not None:
            y_on = (y_lmc + abs(y_lmc))/2
            y_off = (y_lmc - abs(y_lmc))/2
        else:
            y_on, y_off = None, None
        return y_on, y_off

    def get_fdsr(self):
        """ The FDSR mechanism is able to suppress rapidly changed texture
            information and enhance noval contrast change """
        y_on, y_off = self.get_on_off_channels()
        pre_y_on, pre_y_off = STMD(self.last_frame).get_on_off_channels()

        if (y_on is not None) and (y_off is not None):
            try:
                s_on = null_matrix = np.zeros(y_on.shape, dtype=np.float32)
                s_off = null_matrix = np.zeros(y_off.shape, dtype=np.float32)


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

            except Exception as e:
                s_on = None
                s_off = None
                print("ERROR in FDSR", e)
        else:
            raise ValueError("ERROR in FDSR: Y_ON, Y_OFF are empty")

        return s_on, s_off

    def get_sigma(self): # ***
        """ Subtract the filtered signal (s_on, s_off) to the original one
			(y_on, y_off) """
        s_on, s_off = self.get_fdsr()
        y_on, y_off = self.get_on_off_channels()

        """y_on = self.convert_for_display(y_on)
        y_off = self.convert_for_display(y_off)

        s_on = self.convert_for_display(s_on)
        s_off = self.convert_for_display(s_off)"""

        if (s_on is not None) and (y_on is not None):
            f_on = cv2.subtract(y_on, s_on)
            f_off = cv2.subtract(y_off, s_off)
        else:
            f_on, f_off = None, None
            raise ValueError("ERROR in SIGMA: S_ON, S_OFF are empty")
        return f_on, f_off

    def get_hwr(self):
        """ Then we replace all the pixels that are negative with 0. """
        # f_on, f_off = self.get_sigma()
        f_on, f_off = self.get_fdsr()
        f_on = self.convert_for_display(f_on)
        f_off = self.convert_for_display(f_off)
        # ======> HW-R ======
        if f_on is not None:
            f_on = get_max(f_on)
            f_off = get_max(f_off)
        else:
            raise ValueError("ERROR in HW-R: F_ON, F_OFF are empty")
        return f_on, f_off


    # ====================> Lobula LAYER <====================
    def get_li(self):
        """ -> LI plays a significant role in differentiating target motion
            from background motion.
    	    -> The new LIM that considers velocity and motion direction """
        # get f_on, f_off
        f_on, f_off = self.get_hwr()

        # get motion_vectors
        motion_vectors = block_based_motion_estimation(self.frame, self.last_frame, self.block_size, self.search_range)

        # get u, v
        u, v = get_motion_components(motion_vectors)

        # get h
        h = create_convolution_kernel(self.p, self.q, self.a)

        # get u_c, v_c
        u_c, v_c = convolve_uv_with_h(u, v, h)

        # calculate w
        w = calculate_w(u_c, v_c)
        f_on_li = self.k1 * f_on + self.k2 * f_on * w
        f_off_li = self.k1 * f_off + self.k2 * f_off * w

        return f_on_li, f_off_li

    def get_delay(self):
        """ Slight delay on the OFF channel """
        f_on_li, f_off_li = self.get_li()
        if f_off_li is not None:
            lob_off = low_pass_filter(f_off_li, self.tau_delay)
        else:
            raise ValueError("ERROR in DELAY: F_OFF is empty")
        return f_on_li, lob_off

    def get_final_output(self):
        """ Exhibits correlation between ON and OFF channels """
        f_on_li, lob_off = self.get_delay()

        if (f_on_li is not None) and (lob_off is not None):
            output = cv2.multiply(f_on_li, lob_off)
        else:
            output = None
            raise ValueError("ERROR in FINAL OUTPUT: F_ON or F_OFF are empty")

        return output


    # ====================> Other methods <====================
    def convert_for_process(self, frame):
        """ Frame preprocessing """
        # Convert frame to Gray
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert the image to float32 to prevent overflow
        frame = frame.astype(np.float32) / 255.0
        return frame

    def convert_for_display(self, frame):
        """ Prepare frame for display """
        # Convert the image back to the original range (0-255)
        frame = (frame * 255).astype(np.uint8)
        #input = cv2.resize(input, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame


# ====================> Functions <====================
def low_pass_filter(image, tau):
    """ Apply a simple low-pass filter to an image """
    # Apply a simple low-pass filter to the image
    kernel_size = int(2 * np.ceil(2 * tau) + 1)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=tau, sigmaY=tau)
    return blurred_image

def get_max(img):
    """ for HW-R """
    # Compare each pixel value to 0 and get a mask with the result
    mask = cv2.compare(img, 0, cv2.CMP_GT)

    # Create an array with the same shape as the grayscale image and fill it with zeros
    zero_array = np.zeros_like(img)

    # Compute the maximum between the grayscale image and the zero array for each pixel
    max_image = cv2.max(img, zero_array)

    # Set the values in max_image to 0 where the original pixel values were negative
    max_image = cv2.bitwise_and(max_image, max_image, mask=mask)

    return max_image


# ====================> Fonctions for LI ====================
def calculate_sad(block1, block2):
    """ Sum of absolute differences between two block
            or (the matching criteria) -> equayion [6] """
    sad =  np.sum(np.abs(block1 - block2))
    return sad

def find_optimal_motion_vector(current_block, last_frame, i, j, search_range):
    """ To calculate the translation vector (namely the motion vector)
            that minimizes the motion criteria -> equation [17]        """
    min_sad = float('inf')
    optimal_motion_vector = (0, 0)

    block_height, block_width = current_block.shape

    for u in range(-search_range, search_range + 1):
        for v in range(-search_range, search_range + 1):
            shifted_i = np.clip(i + u, 0, last_frame.shape[0] - block_height)
            shifted_j = np.clip(j + v, 0, last_frame.shape[1] - block_width)
            shifted_block = last_frame[shifted_i:shifted_i + block_height, shifted_j:shifted_j + block_width]
            sad = calculate_sad(current_block, shifted_block)

            if sad < min_sad:
                min_sad = sad
                optimal_motion_vector = (u, v)

    return optimal_motion_vector

def block_based_motion_estimation(current_frame, last_frame, block_size, search_range):
    """ This code divides the current frame into blocks of a given size and
        finds the optimal motion vector for each block by searching within
        the last frame. for our case, blok_size = 1.
        So it will find the optical motion for each pixel """
    height, width = current_frame.shape
    num_blocks_height = int(np.ceil(height / block_size))
    num_blocks_width = int(np.ceil(width / block_size))
    motion_vectors = np.zeros((num_blocks_height, num_blocks_width, 2), dtype=int)
    n = 0
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            current_block = current_frame[i:min(i + block_size, height), j:min(j + block_size, width)]
            motion_vectors[i // block_size, j // block_size] = find_optimal_motion_vector(current_block, last_frame, i, j, search_range)
            n = n+1
        print(n/(height*width))

    return motion_vectors

def get_motion_components(motion_vectors):
    """ To get the motion vector matrix U, V """
    U = motion_vectors[:,:,0] # equation 18
    V = motion_vectors[:,:,1] # equation 19
    return U, V

def create_convolution_kernel(p, q, a):
    """ Generate kernel H """
    assert q > 1, "q should be greater than 1 to avoid division by zero"
    b = -a / (4 * (q - 1))

    H = np.zeros((p, q))
    H[0, :] = b
    H[p-1, :] = b
    H[:, 0] = b
    H[:, q-1] = b
    H[p // 2, q // 2] = a

    return H

def convolve_uv_with_h(U, V, H):
    """ Convlolve U and V by H in order to get """
    U_c = convolve2d(U, H, mode='same') # equation 20
    V_c = convolve2d(V, H, mode='same') # equation 21

    return U_c, V_c

"""
The neighborhood defined here, PR, consists of the pixels that are
in a region surrounding the pixel at (i, j). This region seems to be
a rectangular area that extends ro pixels vertically and co pixels
horizontally from (i, j). The region includes the rows from (i-ro) to
(i+ro) and the columns from (j-co) to (j+co).
"""

def calculate_w(U_c, V_c):
    """ Calulate W """
    w = np.sqrt(U_c**2 + V_c**2)
    return w

def lateral_inhibition(F_ON, F_OFF, w, k1, k2):
    """ hen, we implement our new lateral inhibition mechanism
        by multiplying F_ON and F_OFF by w, respectively  """
    F_ON_new = k1 * F_ON + k2 * F_ON * w
    F_OFF_new = k1 * F_OFF + k2 * F_OFF * w
    return F_ON_new, F_OFF_new


# ====================> REmarques ====================
"""
Remarque: On manque de recule.
    Pour les choix de constante notamment k1 et k2.
    Une combinaison en LMC et LI pourrait donnéer des resultats interressante.

Fais:
    -> Calule de LI d'un seul frame prend beaucoup de temps. (blok_size=1)
    Conséquence: Prolème possible:
        * mauvais chois des constantes
        * inssufissance de tests pour avoir assez de recule; (il faur traiter au moins 10 à frame successif).
"""

"""def resize_image(image, scale_percent):

    # Calculate the new dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_dimensions = (width, height)

    # Resize the image
    image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    return image"""
