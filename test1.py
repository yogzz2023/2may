import numpy as np
import matplotlib.pyplot as plt

# Kalman Filter Parameters
F = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0],   # State transition matrix
              [0, 1, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1, 0, 0],   # Velocity components
              [0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],   # Measurement matrix
              [0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0]])
Q = np.eye(9) * 0.01  # Process noise covariance matrix
R = np.eye(4) * 0.1   # Measurement noise covariance matrix

def kalman_filter(x_hat, P, z):
    # Prediction step
    x_hat_minus = np.dot(F, x_hat)
    P_minus = np.dot(np.dot(F, P), F.T) + Q
    
    # Measurement update step
    K = np.dot(np.dot(P_minus, H.T), np.linalg.inv(np.dot(np.dot(H, P_minus), H.T) + R))
    x_hat = x_hat_minus + np.dot(K, (z - np.dot(H, x_hat_minus)))
    P = np.dot((np.eye(9) - np.dot(K, H)), P_minus)
    
    return x_hat, P

# Generate synthetic measurements
sensor_measurements = [
    [[100, 30, 20, 1], [110, 35, 25, 1]],  # Sensor 1 measurements
    [[120, 40, 30, 1], [130, 45, 35, 1]],  # Sensor 2 measurements
    [[105, 32, 22, 2], [115, 37, 27, 2]]   # Additional measurements after 1 time step
]

# Define initial state and covariance
x_hat = np.zeros((9, 1))  # Initial state vector with zeros
P = np.eye(9) * 0.1        # Initial covariance matrix

# Joint Probabilistic Data Association (JPDA) with Kalman Filter
for sensor_idx, measurements in enumerate(sensor_measurements):
    for sensor_measurement in measurements:
        z = np.array(sensor_measurement).reshape(-1, 1)
        x_hat, P = kalman_filter(x_hat, P, z)

# Print final estimated state vector
print("Final estimated state vector (including velocity components):")
print(x_hat.flatten())

# Plot settings
plt.figure(figsize=(10, 5))
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('Sensor Measurements and Estimated Target Positions')

# Plot sensor measurements
for sensor_idx, measurements in enumerate(sensor_measurements):
    for measurement in measurements:
        range_, azimuth, elevation, time = measurement
        plt.scatter(range_, azimuth, label=f"Sensor {sensor_idx + 1} Measurement", color='b' if sensor_idx == 0 else 'r')

# Plot estimated target position
plt.scatter(x_hat[0], x_hat[1], marker='x', color='g', label="Estimated Target Position")

plt.legend()
plt.grid(True)
plt.show()
