import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from scipy.stats import zscore



# Load the simulated accelerometer data

df = pd.read_csv("data/simulated/simulated_accelerometer_data.csv")
#print(df.head()) #prinintg first 5 rows of the dataframe

# Statistical analysis of the simulated data, grouped by orientation and calculating mean ((x+y+z)/3) and standard deviation 

stats = df.groupby('orientation_index')[['true_measurement_x', 'true_measurement_y', 'true_measurement_z']].agg(['mean', 'std'])
#print(stats)

# Calculate the magnitude of the accelerometer readings and describe its statistics

df["magnitude"]=np.sqrt(df["true_measurement_x"]**2+df["true_measurement_y"]**2+df["true_measurement_z"]**2)
#print(df["magnitude"].describe())

# Visualize the raw accelerometer data for each axis

#for axis in ['true_measurement_x', 'true_measurement_y', 'true_measurement_z']:
    
    #plt.plot(df.index, df[axis], label=axis)
    #plt.xlabel("Sample")
    #plt.ylabel("Acceleration [m/s²]")
    #plt.legend()
    #plt.title("Raw accelerometer data")
    #plt.show()

# Moving average smoothing with a window size of 5 samples (i-2, i-1, i, i+1, i+2)

df["x_moving_avg"] = df["true_measurement_x"].rolling(window=5, center=True).mean()
df["y_moving_avg"] = df["true_measurement_y"].rolling(window=5, center=True).mean()
df["z_moving_avg"] = df["true_measurement_z"].rolling(window=5, center=True).mean()

# Define a low-pass Butterworth filter

def butter_lowpass(cutoff,fs,order=3):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = butter(order,normal_cutoff,btype='low',analog=False)
    return b, a

#Apply the low-pass filter to the data

def apply_lowpass(data, cutoff=2.0, fs=50, order=3):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)

# Apply the low-pass filter to each axis of the accelerometer data and store the results in new columns

df["x_filtered"] = apply_lowpass(df["true_measurement_x"])
df["y_filtered"] = apply_lowpass(df["true_measurement_y"])
df["z_filtered"] = apply_lowpass(df["true_measurement_z"])

# Filter values validation by plotting

plt.plot(df.index, df["true_measurement_x"], label="raw")
plt.plot(df.index, df["x_filtered"], label="filtered")

plt.legend()
plt.title("AX raw vs filtered")
#plt.show()

# Outlier detection using Z-score method on the magnitude of the accelerometer data z=(x-mean)/std

df["magnitude_score"]=zscore(df["magnitude"])
outlier=df[np.abs(df["magnitude_score"])>3]
print("Number of outliers detected:", len(outlier))

# Save the processed data to a new CSV file

df.to_csv("data/simulated/processed_accelerometer_data.csv", index=False)