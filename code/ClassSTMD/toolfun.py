import numpy as np

class ToolFun:
    @staticmethod
    def Generalize_Gammakernel(Order, Tau, wide):
        if wide <= 1:
            wide = 2
        Gamma = np.zeros((1, wide))
        for k in range(wide):
            t = k-1
            Gamma[0, k] = (Order*t/Tau)**Order * np.exp(-Order*t/Tau) / \
                (np.math.factorial(Order-1)*Tau)
        Gamma = Gamma / np.sum(Gamma)
        return Gamma

    @staticmethod
    def Generalize_FractionalDerivativeKernel(alpha, wide):
        if wide < 2:
            wide = 2
        kernel_K = np.zeros((1, wide-1))
        kernel_F = np.zeros((1, wide))
        if alpha == 1:
            kernel_K[0, 0] = 1
        elif alpha > 0 and alpha < 1:
            for k in range(wide-1):
                t = k-1
                kernel_K[0, k] = np.exp(-alpha*t/(1-alpha)) / (1-alpha)
            kernel_K = kernel_K / np.sum(kernel_K)
        else:
            raise ValueError('Alpha must be in the range (0,1].')
        kernel_F[0, 0] = kernel_K[0, 0]
        kernel_F[0, 1:-1] = kernel_K[0, 1:] - kernel_K[0, :-1]
        kernel_F[0, -1] = - kernel_K[0, -1]
        return kernel_F

    @staticmethod
    def Generalize_Lateral_InhibitionKernel_W2(KernelSize=15, Sigma1=1.5, Sigma2=3, e=1.0, rho=0, A=1, B=3):
        if KernelSize % 2 == 0:
            KernelSize = KernelSize + 1
        CenX = round(KernelSize/2)
        CenY = round(KernelSize/2)
        X, Y = np.meshgrid(np.arange(1, KernelSize+1), np.arange(1, KernelSize+1))
        ShiftX = X-CenX
        ShiftY = Y-CenY
        Gauss1 = (1/(2*np.pi*Sigma1**2))*np.exp(-(ShiftX**2 + ShiftY**2)/(2*Sigma1**2))
        Gauss2 = (1/(2*np.pi*Sigma2**2))*np.exp(-(ShiftX**2 + ShiftY**2)/(2*Sigma2**2))
        DoG_Filter = Gauss1 - e*Gauss2 - rho
        Positive_Component = np.maximum(DoG_Filter, 0)
        Negative_Component = np.maximum(-DoG_Filter, 0)
        InhibitionKernel_W2 = A*Positive_Component - B*Negative_Component
        return InhibitionKernel_W2

    @staticmethod
    def Generalize_DSTMD_Directional_InhibitionKernel(DSTMD_Directions=8, Sigma1=1.5, Sigma2=3.0):
        if DSTMD_Directions is None:
            DSTMD_Directions = 8
        if Sigma1 is None:
            Sigma1 = 1.5
        if Sigma2 is None:
            Sigma2 = 3.0
        KernelSize = DSTMD_Directions
        Zero_Point_DoG_X1 = -np.sqrt((np.log(Sigma2/Sigma1)*2*Sigma
