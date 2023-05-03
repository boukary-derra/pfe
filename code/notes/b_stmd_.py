import numpy as np
from scipy import ndimage
from skimage import io
import cv2
import scipy.signal


class BasalSTMD_Core:
    def __init__(self):
        self.GaussFilter_SIZE = 3
        self.GaussFilter_SIGMA = 1
        self.Gammakernel_3_Order = 12
        self.Gammakernel_3_Tau = 25
        self.path0 = None
        self.Imagetitle = 'GeneratingDataSet'
        self.Imagetype = '.tif'
        self.StartFrame = 1
        self.EndFrame = 450
        self.SamplingFrequency = None
        self.IsWaitbar = 1
        self.Isvisualize = 0
        self.IsSaveAsVideo = 0
        self.IsRecordOutput = None
        self._init_hidden_properties()

    def _init_hidden_properties(self):
        self.ImageName = None
        self.IMAGE_H = None
        self.IMAGE_W = None
        self.GaussFilter = None
        self.LMCs_len = None
        self.Gammakernel_3 = None
        self.InputState = 0
        self.InhibitionKernel_W2 = None
        self.original_image = None
        self.Input = None
        self.Photoreceptors_Output = None
        self.Cell_Photoreceptors_Output = None
        self.Lamina_Output = None
        self.ON_Channel = None
        self.OFF_Channel = None
        self.Cell_ON_Channel = None
        self.Cell_OFF_Channel = None
        self.Delay_OFF_Channel = None
        self.Correlation_Output = None
        self.Lateral_Inhibition_Output = None
        self.Lobula_Output = None
        self.Cell_Output = None
        self.Direction = None
        self.Cell_Direction = None
        self.Output = None
        self.Gammakernel_3_len = None
        self.IsRecord = 0
        self.NowFrame = 1
        self.Video_Par = ['C:\\Users\\HP\\Desktop', 'test']
        self.H = None
        self.get_ImageName = None
        self.visualize_Threshold = None

    def init_GaussFilter(self):
        self.GaussFilter = ndimage.filters.gaussian_filter(
            self.GaussFilter_SIZE, self.GaussFilter_SIGMA)

    def init_Gammakernel_3(self):
        # The `Generalize_Gammakernel` function is not provided in the original code,
        # assuming it is a function of the `ClassSTMD` class, you need to replace it
        # with an equivalent Python function.
        self.Gammakernel_3 = ClassSTMD.ToolFun.Generalize_Gammakernel(
            self.Gammakernel_3_Order, self.Gammakernel_3_Tau, self.Gammakernel_3_len)

    # Add other methods here as required (Init, getImageName, Read_Image2gray, Retina, Lamina, Medulla, Lobula, RecordOutput, Visualize, Run)
    """ init """
    def init(self):
        # Init Initialize function

        # weakly dependent variable
        if self.Gammakernel_3_len is None:
            self.Gammakernel_3_len = 3 * int(np.ceil(self.Gammakernel_3_Tau))
        if self.LMCs_len is None:
            self.LMCs_len = 2

        # Real-time detection, dead loop
        if self.EndFrame == 0:
            self.StartFrame = 1
            self.Isvisualize = 1
            if self.IsRecordOutput is None:
                self.IsRecordOutput = 0
        elif self.IsRecordOutput is None:
            self.IsRecordOutput = 1

        # init kernel
        self.init_GaussFilter()
        self.init_Gammakernel_3()
        # The `Generalize_Lateral_InhibitionKernel_W2` function is not provided in the original code,
        # assuming it is a function of the `ClassSTMD` class, you need to replace it
        # with an equivalent Python function.
        self.InhibitionKernel_W2 = ClassSTMD.ToolFun.Generalize_Lateral_InhibitionKernel_W2()

        # gets the data set picture size
        self.NowFrame = self.StartFrame
        self.read_image2gray()

        # allocate memory
        self.IMAGE_H, self.IMAGE_W = self.Input.shape

        self.Cell_Photoreceptors_Output = [None] * self.LMCs_len

        self.Cell_OFF_Channel = [None] * self.Gammakernel_3_len

        if self.IsRecordOutput:
            self.Cell_Output = [None] * self.EndFrame

        # instantiate the visualization class and assign the handle
        if self.Isvisualize == 1 or self.IsWaitbar == 1 or self.IsSaveAsVideo == 1:
            class_name = self.__class__.__name__
            # Instantiate the visualization class here.
            # Replace `ClassSTMD.visualization` with the actual visualization class in Python
            self.H = ClassSTMD.visualization(class_name)
            if self.IsSaveAsVideo == 1:
                # Save the visual output as a video
                self.H.IsSaveAsVideo = 1
                self.Isvisualize = 1
                self.H.SavePath = self.Video_Par[0]
                self.H.VideoName = self.Video_Par[1]
            if self.Isvisualize == 1:
                # Instantiate the figure handle class
                self.H.establish_fig_handle()
                if self.visualize_Threshold is not None:
                    self.H.Show_Threshold = self.visualize_Threshold
            if self.IsWaitbar == 1:
                # Instantiate the figure handle class
                self.H.establish_bar_handle()


    """ get image name """
    def get_image_name(self):
        if self.get_ImageName is None:
            self.ImageName = f"{self.path0}\\{self.Imagetitle}{self.NowFrame:04d}{self.Imagetype}"
        else:
            self.ImageName = self.get_ImageName(
                self.path0,
                self.Imagetitle,
                self.Imagetype,
                self.NowFrame
            )


def retina(self):
    # Retina retina layer
    # perform GaussFilter to input
    self.Photoreceptors_Output = scipy.signal.convolve2d(self.Input, self.GaussFilter, mode='same')
    # record the photoreceptors output
    self.Cell_Photoreceptors_Output.pop(0)
    self.Cell_Photoreceptors_Output.append(self.Photoreceptors_Output)
