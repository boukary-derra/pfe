import cv2
import numpy as np
import itertools


frame = cv2.imread("media/complex_frames/frame_0063.jpg")
last_frame = cv2.imread("media/complex_frames/frame_0062.jpg")


frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
"""
frame = cv2.resize(frame, (12, 12))
last_frame = cv2.resize(last_frame, (12, 12))
"""

def get_motion_vectors(search_range):
    h, w = frame.shape
    U = np.zeros((h, w)).astype(np.float64)
    V = np.zeros((h, w)).astype(np.float64)
    d_min = float('inf')
    n = 0
    for (x, y) in itertools.product(range(h), range(w)):
        n = n + 1
        for u in range(-search_range, search_range + 1):
            for v in range(-search_range, search_range + 1):
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

u, v = get_motion_vectors(100)

print(u)
print(np.min(u))
print(np.max(u))
print("\n=========================================")
print(v)
print(np.min(v))
print(np.max(v))
