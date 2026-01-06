import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

G=9.81

df=pd.read_csv("data/simulated/processed_accelerometer_data.csv")
measured_filtered_acceleration=df[["x_filtered","y_filtered","z_filtered"]].values

# Calibration function to minimize the difference between measured and expected gravity magnitude

def calibration_func(params, measured_filtered_acceleration): #params start [0,0,0,1,1,1], in every iteration it will update the params values until it finds the best fit  
  bx, by, bz, sx, sy, sz = params

  b= np.array([bx, by, bz])
  S= np.diag([sx, sy, sz]) # Diagonal matrix for scale factors, assuming no cross-axis sensitivity yet

  calibrated_acceleration = (measured_filtered_acceleration-b)@np.linalg.inv(S).T #T can be ignored for diagonal matrix, it is used for cross-axis sensitivity
  magnitude =  np.linalg.norm(calibrated_acceleration, axis=1)

  return magnitude - G
  
# Perform the calibration using least squares optimization 

result = least_squares(calibration_func, x0=[0,0,0,1,1,1], args=(measured_filtered_acceleration,), method='lm')

# Raw magnitude before calibration

mag_raw = np.linalg.norm(measured_filtered_acceleration, axis=1)
print("Raw magnitude mean:", mag_raw.mean())
print("Raw magnitude std:", mag_raw.std())

# Apply the calibration to the measured data

bx, by, bz, sx, sy, sz = result.x
b= np.array([bx, by, bz])
S= np.diag([sx, sy, sz])
acceleration_calibrated = (measured_filtered_acceleration - b) @ np.linalg.inv(S).T
mag_calibrated = np.linalg.norm(acceleration_calibrated, axis=1)  

print("Calibrated magnitude mean:", mag_calibrated.mean())
print("Calibrated magnitude std:", mag_calibrated.std())

# Plot histograms of raw and calibrated magnitudes for comparison

plt.hist(mag_calibrated, bins=50, color="blue", alpha=0.5, edgecolor="black")
plt.hist(mag_raw, bins=50, color="red", alpha=0.5, edgecolor="black")
#plt.xlim(9.809, 9.811)  # zoom oko gravitacije
plt.title("Calibrated acceleration magnitude")
plt.show()