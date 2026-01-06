#SIMULATION OF ACCELEROMETER DATA WITH BIAS, SCALE MISALIGNMENT, AND NOISE

import numpy as np
import pandas as pd


G=9.81  # Acceleration due to gravity (m/s^2)

# Rotation matrices for different orientations

def rotation_matrix_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
def rotation_matrix_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
def rotation_matrix_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]

    ])

g_world = np.array([0, 0, -G])  # Gravity vector in world frame
I = np.eye(3)

# Accelerometer readings for different orientations

acc_possition_up = I @ g_world                                             
acc_possition_down = rotation_matrix_x(np.deg2rad(180)) @ g_world
acc_possition_left = rotation_matrix_y(np.deg2rad(90)) @ g_world  
acc_possition_right = rotation_matrix_y(np.deg2rad(-90)) @ g_world  
acc_possition_forward = rotation_matrix_x(np.deg2rad(-90)) @ g_world 
acc_possition_backward = rotation_matrix_x(np.deg2rad(90)) @ g_world           

norma_G=np.linalg.norm(acc_possition_up) #used for testing if sqrt(x^2+y^2+z^2)=9.81

# Simulate accelerometer measurements with bias, scale misalignment, and noise

bias = np.array([0.1, -0.05, 0.2])  

scale_misalignment = np.array([
    [1.02, 0.01, -0.02],
    [0.00, 0.98, 0.01],
    [0.01, -0.01, 1.01]
])

noise_std = 0.02  # Standard deviation of noise

# Define function to simulate accelerometer measurement

def acelerometer_measurement(true_acceleration):
    noise = np.random.normal(0, noise_std, size=3)
    ideal_acceleration = true_acceleration@g_world
    measured_acceleration = scale_misalignment @ ideal_acceleration + bias + noise
    return measured_acceleration

# Example orientations in space

orientations=[
    rotation_matrix_x(np.deg2rad(0)),
    rotation_matrix_x(np.deg2rad(90)),
    rotation_matrix_x(np.deg2rad(-90)),
    rotation_matrix_y(np.deg2rad(90)),
    rotation_matrix_y(np.deg2rad(-90)),
    rotation_matrix_z(np.deg2rad(90)),
]

# Generate simulated data and exporting to CSV

samples = []

for i, R in enumerate(orientations):
    for _ in range(200):  # Take 200 samples per orientation
       true_measurement = acelerometer_measurement(R)
       samples.append({
           "orientation_index": i,
           'true_measurement_x': true_measurement[0],
           'true_measurement_y': true_measurement[1],
           'true_measurement_z': true_measurement[2]
          
       })
df= pd.DataFrame(samples)
df.to_csv("data/simulated/simulated_accelerometer_data.csv", index=False)

