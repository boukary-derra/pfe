import os
import time
from parameter_setting import ParameterSetting
from main import Main


import numpy as np
import os
from dbscan import DBSCAN
from time import time
from math import floor
from scipy.io import loadmat, savemat



def parameter_setting():
    pass


def Read_In_Folder_Name():

    # Function description
    # This function reads the file directory and related parameters in TopFunction_TargetTracking_LDTB_Velocity_Size.m

    # Main Function

    # Path of Input Image Sequence
    Parameter_File = {
        'folder0': 'Test-Image-Sequence'
    }

    # Create folder for storing data in RecordedData
    Parameter_File['folder_Global'] = os.path.join('Result/Data-for-DR-FA', Parameter_File['folder0'])

    if not os.path.exists(Parameter_File['folder_Global']):
        os.makedirs(Parameter_File['folder_Global'])

    # Title of Input Image Sequence
    Parameter_File['Imagetitle'] = 'Synthetic-Stimuli'

    # Start and end frame of input image sequence
    Parameter_File['StartRecordFrame'] = 300  # Start frame for recording data
    Parameter_File['StartFrame'] = 1
    Parameter_File['EndFrame'] = 900

    # Call ParameterSetting function to set parameters for the whole process
    ParameterSetting(Parameter_File)

    # Calculate time (Start Point)
    start_time = time.time()

    print('Start Motion Perception...')

    # Call Main function to process Input Image Sequence
    file = os.path.join(Parameter_File['folder_Global'], 'Recorded-Data.mat')
    if not os.path.exists(file):
        Main(Parameter_File)

    # Calculate time (End Point)
    time_train = (time.time() - start_time) / 60  # in minutes

    if time_train < 60:
        print(f"Motion Perception finished, time taken: {time_train:.2f} min")
    else:
        print(f"Motion Perception finished, time taken: {time_train / 60:.2f} hrs")



""" ======================================== """
def Clustering_Detected_Points():
    # Load Max_Operation_DSTMD_Outputs
    if 'Max_Operation_DSTMD_Outputs' not in locals():
        file = os.path.join(Parameter_File['folder_Global'], 'Max_Operation_DSTMD_Outputs.mat')
        Max_Operation_DSTMD_Outputs = loadmat(file)['Max_Operation_DSTMD_Outputs'][0]

    # Generate storage for clustering results
    NumFrame_Clustering = len(Max_Operation_DSTMD_Outputs)
    Clustering_Results = [dict() for _ in range(NumFrame_Clustering)]

    tic = time()
    print('Start Clustering...')

    H_Clustering = 8

    for j in range(NumFrame_Clustering):
        ModelOutputs = Max_Operation_DSTMD_Outputs[j]

        for k in range(1, H_Clustering+1):
            IndX, IndY = np.where(ModelOutputs[:, :, k-1] > DSTMD_Detection_Threshold)
            NIndY = IndY[(IndY > 5) & (IndY < (N_Clustering-5)) & (IndX > 5) & (IndX < (M_Clustering-5))]
            NIndX = IndX[(IndY > 5) & (IndY < (N_Clustering-5)) & (IndX > 5) & (IndX < (M_Clustering-5))]

            if len(NIndY) > 0:
                epsilon = 3
                MinPts = 1
                IDX = DBSCAN(np.column_stack((NIndX, NIndY)), epsilon, MinPts)
            else:
                IDX = [0]

            for direction in range(1, H_Clustering+1):
                if k == direction:
                    Cluster_Num = max(IDX)
                    if Cluster_Num > 0:
                        Cluster_Center = np.zeros((Cluster_Num, 2))
                        for l in range(1, Cluster_Num+1):
                            Clustering_X = NIndX[np.array(IDX) == l]
                            Clustering_Y = NIndY[np.array(IDX) == l]
                            Cluster_Center_X = round(np.mean(Clustering_X))
                            Cluster_Center_Y = round(np.mean(Clustering_Y))
                            Cluster_Center[l-1, :] = [Cluster_Center_X, Cluster_Center_Y]

                        Clustering_Results[j][f'Direction_{direction}_ClusteringPoints'] = Cluster_Center
                        Clustering_Results[j][f'Direction_{direction}_ClusteringPointsNum'] = Cluster_Num
                    else:
                        Clustering_Results[j][f'Direction_{direction}_ClusteringPoints'] = np.array([-1, -1])
                        Clustering_Results[j][f'Direction_{direction}_ClusteringPointsNum'] = 0

    file = os.path.join(Parameter_File['folder_Global'], f'Clustering_Results-Detection-Threshold-{DSTMD_Detection_Threshold}.mat')
    savemat(file, {'StartFrame': StartFrame, 'EndFrame': EndFrame, 'StartRecordFrame': StartRecordFrame, 'Clustering_Results': Clustering_Results, 'DSTMD_Detection_Threshold': DSTMD_Detection_Threshold})

    timeTrain = (time() - tic) / 60  # min
    if timeTrain < 60:
        print(f'Clustering finished, time taken: {timeTrain} min')
    else:
        print(f'Clustering finished, time taken: {timeTrain / 60} hrs
