import numpy as np
import matplotlib.pyplot as plt
from . import *

DSTMD_Detection_Threshold_Range = np.concatenate((np.arange(1e5, 2e4, -1e4), np.arange(1e4, 2e3, -1e3), np.arange(1e3, 1e2, -1e2)))
Num_Detection_Threshold = len(DSTMD_Detection_Threshold_Range)
All_DSTMD_Detection_Rate_Velocity = np.zeros(Num_Detection_Threshold)
All_DSTMD_False_Alarm_Velocity = np.zeros(Num_Detection_Threshold)
All_STMD_Plus_Detection_Rate_Velocity = np.zeros(Num_Detection_Threshold)
All_STMD_Plus_False_Alarm_Velocity = np.zeros(Num_Detection_Threshold)

# Replace these functions with their corresponding Python implementations
Read_In_Folder_Name()
Max_Operation_On_DSTMD_Outputs()

for j_Velocity in range(Num_Detection_Threshold):
    DSTMD_Detection_Threshold = DSTMD_Detection_Threshold_Range[j_Velocity]

    print('=====================================================')
    print(f'Detection Threshold = {DSTMD_Detection_Threshold}')

    # Replace these functions with their corresponding Python implementations
    Clustering_Detected_Points()
    Recording_Target_Trace()
    Show_T1_Neuron_Outputs_Along_Target_Trace()

    # Replace this part with the corresponding Python implementation to load Ground-Truth data
    # if 'Ground_Truth' not in locals():
    #     Ground_Truth = load_ground_truth(Parameter_File.folder0)

    Calculate_Detection_Accuracy_and_False_Alarm()
    All_DSTMD_Detection_Rate_Velocity[j_Velocity] = Detection_Rate_DSTMD
    All_DSTMD_False_Alarm_Velocity[j_Velocity] = False_Alarm_Rate_DSTMD
    All_STMD_Plus_Detection_Rate_Velocity[j_Velocity] = Detection_Rate_STMD_Plus
    All_STMD_Plus_False_Alarm_Velocity[j_Velocity] = False_Alarm_Rate_STMD_Plus

# Remove the Ground_Truth, All_T1_Neuron_Outputs, and Max_Operation_DSTMD_Outputs variables
del Ground_Truth, All_T1_Neuron_Outputs, Max_Operation_DSTMD_Outputs

plt.plot(All_DSTMD_False_Alarm_Velocity, All_DSTMD_Detection_Rate_Velocity, color='r', linestyle='-', marker='>', label='DSTMD', linewidth=1)
plt.plot(All_STMD_Plus_False_Alarm_Velocity, All_STMD_Plus_Detection_Rate_Velocity, color='b', linestyle='-', marker='o', label='STMD Plus', linewidth=1)
plt.legend()
plt.axis([0, 25, 0, 1])
plt.xlabel('False Alarm Rate')
plt.ylabel('Detection Rate')

# Replace this part with the corresponding Python implementation to save data
# file = os.path.join(Parameter_File.folder_Global, 'Detection-Rate-False-Alarm-Rate.mat')
# save_data(file, DSTMD_Detection_Threshold_Range, All_DSTMD_Detection_Rate_Velocity, All_DSTMD_False_Alarm_Velocity, All_STMD_Plus_Detection_Rate_Velocity, All_STMD_Plus_False_Alarm_Velocity)

plt.show()


    
