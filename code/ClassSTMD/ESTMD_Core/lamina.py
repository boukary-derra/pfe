def Lamina(self):
    # Lamina lamina layer

    # Band Pass Filter = Gamma Function1 - Gamma Function2
    Band_Pass_Filter = self.Lamina_Filter.go(self.Cell_Photoreceptors_Output)

    # Inhibition in lamina layer
    self.Lamina_Output = self.Lamina_Inhibition.go(Band_Pass_Filter)
