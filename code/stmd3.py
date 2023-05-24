
""" ============> BROUILLON <============ """

    """def get_fdsr(self):
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

        return s_on, s_off"""


    """def get_photoreceptor(self):
        # Blur effect
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
            l = None
            print("Error in Retina layer / photoreceptor :", e)
        return l

    from scipy.ndimage import convolve"""

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




def calculate_sad(block, last_block):
    """ Sum of absolute differences (sad) between two block
            or (the matching criteria) -> equayion [6] """
    # block2 = cv2.cvtColor(block2, cv2.COLOR_BGR2GRAY)
    sad =  np.sum(np.abs(block - block))
    return sad


"""def find_optimal_motion_vector(current_block, last_frame, i, j, search_range, levels=3):
    # Find the optimal motion vector using hierarchical search.
    min_sad = float('inf')
    optimal_motion_vector = (0, 0)

    block_height, block_width = current_block.shape

    for level in range(levels, 0, -1):
        reduced_current_block = reduce_resolution(current_block, 2**level)
        reduced_last_frame = reduce_resolution(last_frame, 2**level)
        reduced_i = i // 2**level
        reduced_j = j // 2**level
        reduced_search_range = search_range // 2**level

        for u in range(-reduced_search_range, reduced_search_range + 1):
            for v in range(-reduced_search_range, reduced_search_range + 1):
                shifted_i = np.clip(reduced_i + u, 0, reduced_last_frame.shape[0] - reduced_current_block.shape[0])
                shifted_j = np.clip(reduced_j + v, 0, reduced_last_frame.shape[1] - reduced_current_block.shape[1])
                shifted_block = reduced_last_frame[shifted_i:shifted_i + shifted_block.shape[0], shifted_j:shifted_j + shifted_block.shape[1]]
                sad = calculate_sad(reduced_current_block, shifted_block)

                if sad < min_sad:
                    min_sad = sad
                    optimal_motion_vector = (u * 2**level, v * 2**level)

    return optimal_motion_vector"""


def find_optimal_motion_vector(current_block, last_block, i, j, search_range):
    # To calculate the translation vector (namely the motion vector)
            # that minimizes the motion criteria -> equation [17]
    min_sad = float('inf')
    optimal_motion_vector = (0, 0)

    block_height, block_width = current_block.shape

    for u in range(-search_range, search_range + 1):
        for v in range(-search_range, search_range + 1):
            shifted_i = np.clip(i + u, 0, last_block.shape[0] - block_height)
            shifted_j = np.clip(j + v, 0, last_block.shape[1] - block_width)
            shifted_block = last_block[shifted_i:shifted_i + block_height, shifted_j:shifted_j + block_width]
            sad = calculate_sad(current_block, shifted_block)

            if sad < min_sad:
                min_sad = sad
                optimal_motion_vector = (u, v)

    return optimal_motion_vector



"""def block_based_motion_estimation(current_frame, last_frame, block_size, search_range):
    # This code divides the current frame into blocks of a given size and finds the optimal motion vector for each block by searching within
    #    the last frame. for our case, blok_size = 1. So it will find the optical motion for each pixel
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

    return motion_vectors"""

def block_based_motion_estimation(current_frame, last_frame, search_range):
    # This code divides the current frame into blocks of a given size and finds the optimal motion vector for each block by searching within
    #    the last frame. for our case, blok_size = 1. So it will find the optical motion for each pixel
    height, width = current_frame.shape
    motion_vectors = np.zeros((height, width, 2), dtype=int)
    n = 0
    for i in range(0, height):
        for j in range(0, width):
            motion_vectors[i , j] = find_optimal_motion_vector(current_frame, last_frame, i, j, search_range)
        n = n+1
        print(n/(height))

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



"""
The neighborhood defined here, PR, consists of the pixels that are
in a region surrounding the pixel at (i, j). This region seems to be
a rectangular area that extends ro pixels vertically and co pixels
horizontally from (i, j). The region includes the rows from (i-ro) to
(i+ro) and the columns from (j-co) to (j+co).

=====================================================

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