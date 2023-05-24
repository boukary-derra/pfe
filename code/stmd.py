#======= Boukary DERRA (PFE) ==========
# ===== created: 01-02-2023 ===========
# ===== last updated: 09-05-2023 ======

# Modules
import os
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.signal import convolve2d
import numpy as np
import itertools
import cv2



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
        self.fdsr_gaussian_kernel = (11, 11)
        self.tau_delay = 3

        # ======> LI constants ======
        self.k1, self.k2 = 8, 8
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
        gaussian_kernel = self.get_gaussian_kernel()

        try:
            # Apply the gaussian filter to the frame
            buffer_frame = convolve(self.frame, gaussian_kernel, mode='constant')
            return buffer_frame

        except Exception as e:
            print("Error in Retina layer / photoreceptor :", e)
            return None



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
            information and enhance novel contrast change """
        y_on, y_off = self.get_on_off_channels()
        pre_y_on, pre_y_off = STMD(self.last_frame).get_on_off_channels()
        # pre_y_on, pre_y_off = self.get_previous_on_off_channels()

        if (y_on is not None) and (y_off is not None):
            try:
                s_on = np.zeros_like(y_on, dtype=np.float64)
                s_off = np.zeros_like(y_off, dtype=np.float64)

                # Apply Gaussian blur
                blurred_y_on_fast = cv2.GaussianBlur(y_on, self.fdsr_gaussian_kernel, sigmaX=self.tau_fast, sigmaY=self.tau_fast)
                blurred_y_on_slow = cv2.GaussianBlur(y_on, self.fdsr_gaussian_kernel, sigmaX=self.tau_slow, sigmaY=self.tau_slow)
                blurred_y_off_fast = cv2.GaussianBlur(y_off, self.fdsr_gaussian_kernel, sigmaX=self.tau_fast, sigmaY=self.tau_fast)
                blurred_y_off_slow = cv2.GaussianBlur(y_off, self.fdsr_gaussian_kernel, sigmaX=self.tau_slow, sigmaY=self.tau_slow)

                s_on = np.where(y_on > pre_y_on, blurred_y_on_fast, blurred_y_on_slow)
                s_off = np.where(y_off > pre_y_off, blurred_y_off_fast, blurred_y_off_slow)

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

        if (s_on is not None) and (y_on is not None):
            f_on = cv2.subtract(y_on, s_on)
            f_off = cv2.subtract(y_off, s_off)
        else:
            f_on, f_off = None, None
            raise ValueError("ERROR in SIGMA: S_ON, S_OFF are empty")
        return f_on, f_off

    def get_hwr(self):
        """ Then we replace all the pixels that are negative with 0. """
        f_on, f_off = self.get_sigma()

        # ======> HW-R ======
        if f_on is not None:
            f_on = max_with_zero(f_on)
            f_off = max_with_zero(f_off)
        else:
            raise ValueError("ERROR in HW-R: F_ON, F_OFF are empty")

        """f_on=f_on.astype(np.float32)
        f_off=f_off.astype(np.float32)"""
        return f_on, f_off


    # ====================> Lobula LAYER <====================
    # Find the optimal motion vector using
    # * hierarchical search.
    # *Three Step Search (TSS);
    # *the Diamond Search algorithms.
    def get_motion_vector_matrix(self):
        h, w = self.frame.shape
        U = np.zeros((h, w))
        V = np.zeros((h, w))
        d_min = float('inf')
        n = 0
        for (x, y) in itertools.product(range(h), range(w)):
            n = n + 1
            for u in range(-self.search_range, self.search_range + 1):
                for v in range(-self.search_range, self.search_range + 1):
                    # print(frame[i, j])
                    try:
                        d = abs(frame[x, y] - last_frame[x+u, y+v])
                    except: d = None
                    if (d is not None) and d < d_min:
                        d_min = d
                        # motion_vector = (u, v)
                        U[x, y]=u
                        V[x, y]=v
            print(n/(h*w))
        return U, V

    def create_convolution_kernel(self):
        a, p, q = self.a, self.p, self.q
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

    def get_velocity_difference(self):
        """ Convlolve U and V by H in order to get """
        U, V = self.get_motion_vector_matrix()
        H = self.create_convolution_kernel()
        Uc = convolve2d(U, H, mode='same') # equation 20
        Vc = convolve2d(V, H, mode='same') # equation 21
        return Uc, Vc

    def get_w(self):
        """ Calulate W """
        Uc, Vc = self.get_velocity_difference()
        w = np.sqrt(Uc**2 + Vc**2)
        return w

    def get_li(self):
        """ hen, we implement our new lateral inhibition mechanism
            by multiplying F_ON and F_OFF by w, respectively  """
        w = self.get_w()
        k1, k2 = self.k1, self.k2
        f_on, f_off = self.get_hwr()

        f_on_ = k1 * F_ON + k2 * F_ON * w
        f_off_ = k1 * F_OFF + k2 * F_OFF * w

        return f_on_, f_off_

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

        output = normalize_image(output)
        # output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        return output


    # ====================> Other methods <====================
    def convert_for_process(self, frame):
        """ Frame preprocessing """
        # Convert frame to Gray
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Convert the image to float32 to prevent overflow
        # frame = frame.astype(np.float32) / 255.0
        frame = frame.astype(np.float64)
        return frame

    def convert_for_display(self, frame):
        """ Prepare frame for display """
        # Convert the image back to the original range (0-255)
        # frame = (frame * 255).astype(np.uint8)
        #input = cv2.resize(input, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame


    # ====================> Show methods <====================
    def show_photoreceptor(self):
        show_output(self.get_photoreceptor(), "1_photoreceptor")

    def show_lipetz_transformation(self):
        output = self.get_lipetz_transformation()
        # output = (output * 255).astype(np.uint8)
        output = normalize_image(output)
        show_output(output, "2_lipetz_transformation")

    def show_low_pass_filter(self):
        output = self.get_low_pass_filter()
        # output = (output * 255).astype(np.uint8)
        output = normalize_image(output)
        show_output(output, "3_low pass filter")

    def show_lmc(self):
        output = self.get_lmc()
        # output = (output * 255).astype(np.uint8)
        output = normalize_image(output)
        show_output(output, "4_lmc")

    def show_on_off_channels(self):
        on, off = self.get_on_off_channels()
        # on = (on * 255).astype(np.uint8)
        # off = (off * 255).astype(np.uint8)
        on = normalize_image(on)
        off = normalize_image(off)
        show_output(on, "5_on_chanel")
        show_output(off, "5_off_channel")

    def show_fdsr(self):
        on, off = self.get_fdsr()
        """on = (on * 255).astype(np.uint8)
        off = (off * 255).astype(np.uint8)"""
        on = normalize_image(on)
        off = normalize_image(off)
        show_output(on, "6_on_fdsr")
        show_output(off, "6_off_fdsr")

    def show_sigma(self):
        on, off = self.get_sigma()
        """on = (on * 255).astype(np.uint8)
        off = (off * 255).astype(np.uint8)"""
        on = normalize_image(on)
        off = normalize_image(off)
        show_output(on, "7_on_sigma")
        show_output(off, "7_off_sigma")

    def show_hwr(self):
        on, off = self.get_hwr()
        on = normalize_image(on)
        off = normalize_image(off)
        show_output(on, "8_on_hwr")
        show_output(off, "8_off_hwr")

    def show_motion_vector_matrix(self):
        u, v = self.get_motion_vector_matrix()
        u = normalize_image(u)
        v = normalize_image(v)
        show_output(u, "21_matrix_U")
        show_output(v, "21_matrix_V")

    def show_velocity_difference(self):
        u, v = self.get_velocity_difference()
        u = normalize_image(u)
        v = normalize_image(v)
        show_output(u, "22_matrix_Uc")
        show_output(v, "22_matrix_Vc")

    def show_w(self):
        output = self.get_w()
        # output = (output * 255).astype(np.uint8)
        output = normalize_image(output)
        show_output(output, "23_matrix_W")

    def show_li(self):
        on, off = self.get_li()
        on = normalize_image(on)
        off = normalize_image(off)
        show_output(on, "9_on_li")
        show_output(off, "9_off_li")

    def show_delay(self):
        on, off = self.get_delay()
        on = normalize_image(on)
        off = normalize_image(off)
        show_output(on, "10_on_delay")
        show_output(off, "10_off_delay")

    def show_final_output(self):
        output = self.get_final_output()
        show_output(output, "11.final_output")

# ====================> Functions <====================
def low_pass_filter(image, tau):
    """ Apply a simple low-pass filter to an image """
    # Apply a simple low-pass filter to the image
    kernel_size = int(2 * np.ceil(2 * tau) + 1)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=tau, sigmaY=tau)
    return blurred_image


def max_with_zero(image):
    # Create a zero matrix with the same shape as the input image
    zero_matrix = np.zeros_like(image)

    # Get the maximum between 0 and each pixel value of the image
    max_image = cv2.max(image, zero_matrix)

    return max_image

# ====================> REmarques ====================

def show_output(output, name=''):
    file = os.path.join("result", name+'_result.jpg')

    print("\n============================ " + name + " ===============================\n ")
    print(output)

    # output = n_to_width(output, 1920)
    # cv2.imshow(name, output)
    cv2.imwrite(file, output)

def normalize_image(img):
    # epsilon = 1e-6
    # Find the minimum and maximum values in the image
    min_val = np.min(img)
    max_val = np.max(img)

    # Normalize the image so its values range from 0 to 1
    if max_val != min_val:
        img_normalized = (img - min_val) / ((max_val - min_val))
    else:
        img_normalized = np.zeros_like(img, dtype=np.float64)


    # Convert to uint8
    img_uint8 = (img_normalized * 255).astype(np.uint16)

    # Convert to BGR
    # output = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    return img_uint8
