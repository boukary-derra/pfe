import numpy as np
from scipy.signal import convolve2d

class Lamina_Lateral_Inhibition:
    def __init__(self, size_W1=[15,15,7], lambda1=3, lambda2=9, sigma2=1.5, sigma3=None):
        self.size_W1 = size_W1
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sigma2 = sigma2
        if sigma3 is None:
            self.sigma3 = 2 * sigma2
        else:
            self.sigma3 = sigma3
        self.init_ESTMD_W1()

    def init_ESTMD_W1(self):
        G_sigma2 = np.outer(
            np.exp(-0.5 * np.power(np.linspace(-self.size_W1[0]//2, self.size_W1[0]//2, self.size_W1[0]), 2) / np.power(self.sigma2, 2)),
            np.exp(-0.5 * np.power(np.linspace(-self.size_W1[1]//2, self.size_W1[1]//2, self.size_W1[1]), 2) / np.power(self.sigma2, 2))
        )
        G_sigma3 = np.outer(
            np.exp(-0.5 * np.power(np.linspace(-self.size_W1[0]//2, self.size_W1[0]//2, self.size_W1[0]), 2) / np.power(self.sigma3, 2)),
            np.exp(-0.5 * np.power(np.linspace(-self.size_W1[1]//2, self.size_W1[1]//2, self.size_W1[1]), 2) / np.power(self.sigma3, 2))
        )
        temp = G_sigma2 - G_sigma3
        self.W_S_P = np.maximum(temp, 0)
        self.W_S_N = np.maximum(-temp, 0)

        t = np.arange(self.size_W1[2])
        self.W_T_P = np.exp(-t/self.lambda1) / self.lambda1
        self.W_T_N = np.exp(-t/self.lambda2) / self.lambda2

        self.Cell_BP_W_S_P = [None] * self.size_W1[2]
        self.Cell_BP_W_S_N = [None] * self.size_W1[2]

    def go(self, Input):
        self.Cell_BP_W_S_P.pop(0)
        self.Cell_BP_W_S_P.append(convolve2d(Input, self.W_S_P, mode='same'))
        self.Cell_BP_W_S_N.pop(0)
        self.Cell_BP_W_S_N.append(convolve2d(Input, self.W_S_N, mode='same'))

        Lateral_Inhibition_Output = np.sum([bp * w for bp, w in zip(self.Cell_BP_W_S_P, self.W_T_P)], axis=0) + \
                                    np.sum([bp * w for bp, w in zip(self.Cell_BP_W_S_N, self.W_T_N)], axis=0)

        return Lateral_Inhibition_Output
