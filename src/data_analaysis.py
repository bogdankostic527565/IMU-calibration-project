import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Load the simulated accelerometer data

df = pd.read_csv("data/simulated/simulated_accelerometer_data.csv")
print(df.head()) #prinintg first 5 rows of the dataframe

# Statistical analysis of the simulated data, grouped by orientation and calculating mean ((x+y+z)/3) and standard deviation 

stats = df.groupby('orientation_index')[['true_measurement_x', 'true_measurement_y', 'true_measurement_z']].agg(['mean', 'std'])
print(stats)

# Calculate the magnitude of the accelerometer readings and describe its statistics

df["magnituted"]=np.sqrt(df["true_measurement_x"]**2+df["true_measurement_y"]**2+df["true_measurement_z"]**2)
print(df["magnituted"].describe())

# Visualize the raw accelerometer data for each axis

for axis in ['true_measurement_x', 'true_measurement_y', 'true_measurement_z']:
    
    plt.plot(df.index, df[axis], label=axis)
    plt.xlabel("Sample")
    plt.ylabel("Acceleration [m/s²]")
    plt.legend()
    plt.title("Raw accelerometer data")
    plt.show()

df["ax_smooth"] = df["ax"].rolling(window=5, center=True).mean()
df["ay_smooth"] = df["ay"].rolling(window=5, center=True).mean()
df["az_smooth"] = df["az"].rolling(window=5, center=True).mean()