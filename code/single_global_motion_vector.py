import numpy as np
import cv2


# Step 1: Define a function to calculate the SAD for a given motion vector (u, v)
def calculate_sad(current_frame, last_frame, u, v):
    """ Return the matching criteria """
    shifted_frame = np.roll(last_frame, (v, u), axis=(0, 1))
    sad = np.sum(np.abs(current_frame - shifted_frame))
    return sad

def find_optimal_motion_vector(current_frame, last_frame, search_range):
    min_sad = float('inf')
    optimal_motion_vector = (0, 0)

    for u in range(-search_range, search_range + 1):
        for v in range(-search_range, search_range + 1):
            sad = calculate_sad(current_frame, last_frame, u, v)

            if sad < min_sad:
                min_sad = sad
                optimal_motion_vector = (u, v)


    return optimal_motion_vector

# Load the current frame and the last frame as grayscale images
current_frame = cv2.imread('media/test_100.tif', cv2.IMREAD_GRAYSCALE)
last_frame = cv2.imread('media/test_200.tif', cv2.IMREAD_GRAYSCALE)

# Set the search range R
search_range = 8

# Find the optimal motion vector
optimal_motion_vector = find_optimal_motion_vector(current_frame, last_frame, search_range)
print("Optimal motion vector (u', v'):", optimal_motion_vector)
