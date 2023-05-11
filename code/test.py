import cv2
import numpy as np

# assuming B and C are numpy arrays (which is the data type used by cv2 for images)
# let's create some sample data for the sake of demonstration

B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
C = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float32)

# subtract C from B
A = cv2.subtract(B, C)

print(B)
print("====================\n")
print(C)
print("=====================\n")
print(A)
