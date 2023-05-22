from stmd import STMD
import cv2
import numpy as np
import scipy.ndimage as ndimage
import os

"""
frame = cv2.imread("media/simple_frames/frame_0063.jpg")
last_frame = cv2.imread("media/simple_frames/frame_0062.jpg")

"""

frame = cv2.imread("media/complex_frames/frame_0063.jpg")
last_frame = cv2.imread("media/complex_frames/frame_0062.jpg")


model = STMD(frame, last_frame)
model.show_photoreceptor()
model.show_lipetz_transformation()
model.show_low_pass_filter()
model.show_lmc()
model.show_on_off_channels()
model.show_fdsr()
model.show_sigma()
model.show_hwr()
model.show_li()
# model.show_delay()
# model.show_final_output()

cv2.waitKey(0)
cv2.destroyAllWindows()


#show_output(test_image, "test ttt output")


"""
def convolve_image_with_gaussian_mask(image, mask):
    # Convert the image to float32 to prevent overflow
    #image_float = image.astype(np.float32) / 255.0

    # Convolve the image with the Gaussian mask
    convolved_image = ndimage.convolve(image, mask)

    # Convert the image back to the original range (0-255)
    #convolved_image = (convolved_image * 255).astype(np.uint8)

    return convolved_image

test_image = convolve_image_with_gaussian_mask(stmd.input, stmd.gaussian_mask)
show_output(test_image, "Test ttt")




def low_pass_filter(image, tau):
    # Apply a simple low-pass filter to the image
    kernel_size = int(2 * np.ceil(2 * tau) + 1)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=tau, sigmaY=tau)
    return blurred_image

def modified_lipetz_transformation(image, u, tau):
    # Convert the image to float32 to prevent overflow
    image_float = image.astype(np.float32) / 255.0

    # Calculate the low-pass filtered version of the image
    low_pass_image = low_pass_filter(image_float, tau)

    # Apply the modified Lipetz transformation
    numerator = np.power(image_float, u)
    denominator = np.power(image_float, u) + np.power(low_pass_image, u)
    transformed_image = np.divide(numerator, denominator)

    # Convert the image back to the original range (0-255)
    transformed_image = (transformed_image * 255).astype(np.uint8)

    return transformed_image

# Load an image
input_image = cv2.imread('media/image2.jpg', cv2.IMREAD_COLOR)

# Apply the modified Lipetz transformation
u = 0.7
tau = 3
output_image = modified_lipetz_transformation(input_image, u, tau)

# Save the transformed image
cv2.imwrite('output_image.jpg', output_image)

# Display the input and transformed images
cv2.imshow('Input Image', input_image)
cv2.imshow('Modified Lipetz Transformed Image', output_image)
"""
