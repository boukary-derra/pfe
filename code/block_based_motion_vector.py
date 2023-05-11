import numpy as np
from scipy.signal import convolve2d
import cv2

def calculate_sad(block1, block2):
    return np.sum(np.abs(block1 - block2))

def find_optimal_motion_vector(current_block, last_frame, i, j, search_range):
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
    height, width = current_frame.shape
    num_blocks_height = int(np.ceil(height / block_size))
    num_blocks_width = int(np.ceil(width / block_size))
    motion_vectors = np.zeros((num_blocks_height, num_blocks_width, 2), dtype=int)

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            current_block = current_frame[i:min(i + block_size, height), j:min(j + block_size, width)]
            motion_vectors[i // block_size, j // block_size] = find_optimal_motion_vector(current_block, last_frame, i, j, search_range)

    return motion_vectors



    return motion_vectors

# Load the current frame and the last frame as grayscale images
current_frame = cv2.imread('media/test_100.tif', cv2.IMREAD_GRAYSCALE)
last_frame = cv2.imread('media/test_200.tif', cv2.IMREAD_GRAYSCALE)

# Set the block size and the search range
block_size = 1
search_range = 8
# Perform block-based motion estimation
# motion_vectors = block_based_motion_estimation(current_frame, last_frame, block_size, search_range)

"""for i in range(motion_vectors.shape[0]):
    for j in range(motion_vectors.shape[1]):
        print(f"Motion vector for block at ({i}, {j}):", motion_vectors[i, j])"""

def get_motion_components(motion_vectors):
    U = motion_vectors[:,:,0]
    V = motion_vectors[:,:,1]
    return U, V

# U, V = get_motion_components(motion_vectors)
"""print(U.shape)
print("========================================================================")
print(U)
print("========================================================================")
print(V)"""



def create_convolution_kernel(p, q, a):
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
    U_c = convolve2d(U, H, mode='same')
    V_c = convolve2d(V, H, mode='same')

    return U_c, V_c

# Assuming you have U and V matrices

p, q = 5, 5  # change to your desired size
a = 1  # change to your desired value

H = create_convolution_kernel(p, q, a)
# U_c, V_c = convolve_uv_with_h(U, V, H)

print(H.shape)
print(H)

def calculate_w(U_c, V_c):
    w = np.sqrt(U_c**2 + V_c**2)
    return w

# w = calculate_w(U_c, V_c)

def lateral_inhibition(F_ON, F_OFF, w, k1, k2):
    F_ON_new = k1 * F_ON + k2 * F_ON * w
    F_OFF_new = k1 * F_OFF + k2 * F_OFF * w
    return F_ON_new, F_OFF_new

# Assuming you have matrices F_ON and F_OFF, and constants k1 and k2
# F_ON_new, F_OFF_new = lateral_inhibition(F_ON, F_OFF, w, k1, k2)
