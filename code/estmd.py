#======= Boukary DERRA ==========

# Modules
import numpy as np
import itertools
import cv2


class ESTMD:
    def __init__(self, pre_frame, frame, t=1):
        # Convert input frame to Gray Syle
        self.pre_frame = pre_frame
        self.t = t
        cv2.imshow("Real frame", frame)
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray style", self.frame)
        self.w = np.array([1/16, 1/8, 1/16, 1/8, 1/4, 1/8, 1/16, 1/8, 1/16]).reshape(3, 3)
        self.u = 0.7
        self.tau_1 = 3
        self.tau_2 = 3
        self.tau_3 = 3
        self.tau_fast = 3
        self.tau_slow = 3
        self.tau_5 = 3
        self.const_1 = 5
        self.const_2 = 12
        self.const_3 = 8
        self.const_4 = 12
        self.const_5 = 8



    """ =============================== TOOLS =============================="""
    def get_shape(self):
        """ get the size of the current frame """
        return self.frame.shape

    def get_equa_diff_solution(self, input, t, tau, const):
        """ Solutions of the equation [4, 5, 6, 10, 11, 28] """
        return const*np.exp((-1/tau)*t)+input


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
            y_lmc_buffer = np.zeros((m, n))
            y_on_buffer = np.zeros((m, n))
            y_off_buffer = np.zeros((m, n))
            for (i, j) in itertools.product(range(m), range(n)):
                x = self.frame[i, j]

                # equation [6]
                x_lmc = self.get_equa_diff_solution(x, self.t, self.tau_3, self.const_3)

                # equation [7]
                y_lmc = x - x_lmc
                y_lmc_buffer[i, j] = y_lmc

                # equation [8]
                y_on_buffer[i, j] = (y_lmc + abs(y_lmc))/2

                # equation [9]
                off_buffer[i, j] = (y_lmc - abs(y_lmc))/2

            self.y_lmc  = y_lmc_buffer.astype(np.uint8)
            self.y_on = y_on_buffer.astype(np.uint8)
            self.y_off = y_off_buffer.astype(np.uint8)

        except Exception as e:
            print("Error in Lamina layer / high pass filter :", e)
            self.y_on, self.y_off = None, None

        return (self.y_on, self.y_off)


    """ ======================= Medulla Layer =============================="""
    def FDSR(self):
        """ The FDSR mechanism is able to suppress rapidly changed texture
        information and enhance noval contrast change """

        (pre_y_on, pre_y_off) = ESTMD(self.pre_frame, t-1).FDSR()
        self.high_pass_filter()
        (m, n) = self.get_shape()

        try:
            s_on_buffer = np.zeros((m, n))
            s_off_buffer = np.zeros((m, n))
            for (i, j) in itertools.product(range(m), range(n)):

                y_on = self.y_on[i, j]
                # equation [10]
                if y_on > pre_y_on:
                    s_on = self.get_equa_diff_solution(y_on, self.t, self.tau_fast, self.const_4)
                else:
                    s_on = self.get_equa_diff_solution(y_on, self.t, self.tau_slow, self.const_4)
                s_on_buffer[i, j] = s_on

                y_off = self.y_off[i, j]
                # equation [11]
                if y_off > pre_y_off:
                    s_off = self.get_equa_diff_solution(y_off, self.t, self.tau_fast, self.const_4)
                else:
                    s_off = self.get_equa_diff_solution(y_off, self.t, self.tau_slow, self.const_4)
                s_off_buffer[i, j] = s_off


            self.s_on = s_on_buffer.astype(np.uint8)
            self.s_off = s_on_buffer.astype(np.uint8)

        except Exception as e:
            print("Error in Medulla layer / FDSR :", e)


    def sigma_et_HWR(self):
        (pre_y_on, pre_y_off) = ESTMD(self.pre_frame, t-1).FDSR()
        self.FDSR()
        (m, n) = self.get_shape()

        try:
            f_on_buffer = np.zeros((m, n))
            f_off_buffer = np.zeros((m, n))
            for (i, j) in itertools.product(range(m), range(n)):

                y_on = self.y_on[i, j]
                s_on = self.s_on[i, j]
                # equation [12]
                f_on_buffer[i, j] = max(0, y_on - s_on)

                y_off = self.y_off[i, j]
                s_off = self.s_off[i, j]
                # equation [13]
                f_off_buffer[i, j] = max(0, y_off - s_off)

            self.f_on = f_on_buffer.astype(np.uint8)
            self.f_off = f_off_buffer.astype(np.uint8)

        except Exception as e:
            print("Error in Medulla layer / Sigma or HW-R :", e)




    """ ======================= Lobula Layer ============================"""
    def LI(self):
        (pre_y_on, pre_y_off) = ESTMD(self.pre_frame, t-1).FDSR()
        self.FDSR()
        (m, n) = self.get_shape()

        try:
            buffer = np.zeros((m, n))
            buffer = np.zeros((m, n))
            for (i, j) in itertools.product(range(m), range(n)):
                pass

        except Exception as e:
            print("Error in Lobula Layer / LI :", e)

    def delay(self):
        """ Slight delay on the OFF channel """
        self.LI()
        (m, n) = self.get_shape()

        try:
            buffer = np.zeros((m, n))
            for (i, j) in itertools.product(range(m), range(n)):
                # equation [28]
                lob_off = self.get_equa_diff_solution(f_off, self.t, self.tau_5, self.const_5)
                buffer[i, j] = lob_off
            self.f_off = buffer.astype(np.uint8)
        except Exception as e:
            print("Error in Lobula Layer / delay :", e)

    def final_output(self):
        """ Exhibits correlation between ON and OFF channels """
        self.LI()
        (m, n) = self.get_shape()

        try:
            buffer = np.zeros((m, n))
            for (i, j) in itertools.product(range(m), range(n)):
                # equation [29]
                output = self.f_on[i, j]*self.f_off[i, j]
                buffer[i, j] = output
            self.output = buffer.astype(np.uint8)
            cv2.imshow("Final output", self.output)
        except Exception as e:
            print("Error in Lobula Layer / final output :", e)
