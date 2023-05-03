from stmd import STMD
import cv2
import numpy as np
import scipy.ndimage as ndimage
import os

frame = cv2.imread("media/frame_6.jpg")
pre_frame = cv2.imread("media/frame_5.jpg")


def show_output(output, name=''):
    file = os.path.join("result", name+'.jpg')
    print("\n============================ " + name + " ===============================\n ")
    print(output)

    # output = n_to_width(output, 1920)
    cv2.imshow(name, output)
    cv2.imwrite(file, output)



stmd = STMD(frame)
stmd.LMC()
output1 = stmd.gaussian_kernel
output2 = stmd.photoreceptor_output
output3 = stmd.lipetz_transformation_output
output4 = stmd.low_pass_filter_output
output5 = stmd.get_std() # lmc output
output5_ = stmd.convert_for_display(output5)

frame = output5
stmd2 = STMD(pre_frame)
stmd2.LMC()
pre_frame = stmd2.get_std()
output6 = stmd.get_delta(frame, pre_frame)
output6 = stmd.convert_for_display(output6)


show_output(output1, "gaussian_kernel")
show_output(output2, "photoreceptor")
show_output(output3, "lipetz_transformation")
show_output(output4, "low_pass_filter")
show_output(output5_, "LMC")
show_output(output6, "DELTA")



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
