from ClassSTMD import basalSTMD_Core, Lamina_Lateral_Inhibition, Gamma_Filter

class ESTMD_Core(basalSTMD_Core):
    def __init__(self):
        super().__init__()
        self.Lamina_Inhibition = Lamina_Lateral_Inhibition()

    def Init(self):
        if not hasattr(self, 'Gammakernel_1_len'):
            self.Gammakernel_1_len = 3 * math.ceil(self.Gammakernel_1_Tau)
        if not hasattr(self, 'Gammakernel_2_len'):
            self.Gammakernel_2_len = 3 * math.ceil(self.Gammakernel_2_Tau)
        if not hasattr(self, 'LMCs_len'):
            self.LMCs_len = max(self.Gammakernel_1_len, self.Gammakernel_2_len)

        super().Init()

        self.Lamina_Filter = Gamma_Filter(
            self.Gammakernel_1_Order,
            self.Gammakernel_1_Tau,
            self.Gammakernel_1_len,
            self.Gammakernel_2_Order,
            self.Gammakernel_2_Tau,
            self.Gammakernel_2_len
        )

    def Lamina(self):
        pass

    def Medulla(self):
        pass

    def Lobula(self):
        pass
