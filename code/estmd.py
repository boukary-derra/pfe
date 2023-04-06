#======= Boukary DERRA ==========

# Modules
import numpy as np
import itertools
import cv2


class ESTMD:
    def __init__(self, frame, t=1):
        # Convert input frame to Gray Syle
        self.t = t
        cv2.imshow("Real frame", frame)
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray style", self.frame)
        self.w = np.array([1/16, 1/8, 1/16, 1/8, 1/4, 1/8, 1/16, 1/8, 1/16]).reshape(3, 3)
        self.u = 0.7
        self.tau_1 = 3
        self.tau_2 = 3
        self.tau_3 = 3
        self.const_1 = 5
        self.const_2 = 12
        self.const_3 = 8
        self.on_channel = np.zeros(self.get_shape())
        self.off_channel = np.zeros(self.get_shape())


    """ =============================== TOOLS =============================="""
    def get_shape(self):
        """ get the size of the current frame """
        return self.frame.shape

    def get_equa_diff_solution(self, input, t, tau, const):
        """ Solutions of the equation [4, 5, 6, 10, 11, 28] """
        return const*np.exp(2*(-1/tau)*t)+input


    """ ================= Retina Layer ========================="""
    def photoreceptor(self):
        """ Blur effect"""
        # input frame size (m: with, n: height)
        (m, n) = self.get_shape()

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
                            l += self.w[u, v] * self.frame[i+u, j+v]
                        except: pass

                 # fill output frame element by element
                buffer_frame[i, j] = l

            # convert output elements to np.uint8
            self.frame  = buffer_frame.astype(np.uint8)
            cv2.imshow("1. Photoreceptor frame", self.frame)
        except Exception as e:
            print("Error in Retina layer / photoreceptor :", e)

    def lipetz_transformation(self):
        """ transform the input luminance to membrane potential """
        self.photoreceptor()
        (m, n) = self.get_shape()

        try:
            buffer_frame = np.zeros((m, n))
            for (i, j) in itertools.product(range(m), range(n)):
                l = self.frame[i, j]
                # equation [4]
                lc = self.get_equa_diff_solution(l, self.t, self.tau_1, self.const_1)
                # equation [3]
                p = l**self.u/(l**self.u + lc**self.u)
                # Normalisation pour que le frame soit observable
                p = p*(255**(1/self.u))
                buffer_frame[i, j] = p
            self.frame  = buffer_frame.astype(np.uint8)
            cv2.imshow("2. Lipetz transformation", self.frame)

        except Exception as e:
            print("Error in Retina layer / lipetz transformation :", e)


    """ ======================= Lamina Layer ==============================="""
    def low_pass_filter(self):
        """ Slight delay """
        self.lipetz_transformation()
        (m, n) = self.get_shape()

        try:
            buffer_frame = np.zeros((m, n))
            for (i, j) in itertools.product(range(m), range(n)):
                p = self.frame[i, j]
                # equation [5]
                x = self.get_equa_diff_solution(p, self.t, self.tau_2, self.const_2)
                buffer_frame[i, j] = x
            self.frame  = buffer_frame.astype(np.uint8)
            cv2.imshow("3. Low pass filter", self.frame)

        except Exception as e:
            print("Error in Lamina layer / low pass filter :", e)

    def high_pass_filter(self):
        """ Remove redundant information; Maximize information transmission """
        self.low_pass_filter()
        (m, n) = self.get_shape()

        try:
            buffer_frame = np.zeros((m, n))
            for (i, j) in itertools.product(range(m), range(n)):
                x = self.frame[i, j]
                # equation [6]
                x_lmc = self.get_equa_diff_solution(x, self.t, self.tau_3, self.const_3)
                # equation [7]
                y_lmc = x - x_lmc
                buffer_frame[i, j] = y_lmc
            self.frame  = buffer_frame.astype(np.uint8)
            cv2.imshow("4. High pass filter", self.frame)

        except Exception as e:
            print("Error in Lamina layer / high pass filter :", e)


    """ ======================= Medulla Layer =============================="""
    def FDSR(self):
        """ Separate LMCs' output to ON and OFF channels """
        self.high_pass_filter()
        (m, n) = self.get_shape()

        try:
            on_buffer = np.np.zeros((m, n))
            off_buffer = np.np.zeros((m, n))
            for (i, j) in itertools.product(range(m), range(n)):
                y_lmc = self.frame[i, j]
                # equation [8]
                on_buffer[i, j] = (y_lmc + abs(y_lmc))/2
                # equation [9]
                off_buffer[i, j] = (y_lmc - abs(y_lmc))/2
            self.on_channel = on_buffer.astype(np.uint8)
            self.off_channel = off_buffer.astype(np.uint8)
        except Exception as e:
            print("Error in Medulla layer / FDSR :", e)

    def signal(self):
        pass

    def HW_R(self):
        pass


    """ ======================= Lobula Layer ============================"""
    def delay(self):
        """ Slight delay on the OFF channel """
        pass

    def final_output(self):
        """ Exhibits correlation between ON and OFF channels """
        pass
