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

# Parse the provided data
data = """20665.41 178.8938 1.7606 21795.857
20666.14 178.9428 1.7239 21796.389
20666.49 178.8373 1.71 21796.887
20666.46 178.9346 1.776 21797.367
20667.39 178.9166 1.8053 21797.852
20679.63 178.8026 2.3944 21798.961
20668.63 178.8364 1.7196 21799.494
20679.73 178.9656 1.7248 21799.996
20679.9 178.7023 1.6897 21800.549
20681.38 178.9606 1.6158 21801.08
33632.25 296.9022 5.2176 22252.645
33713.09 297.0009 5.2583 22253.18
33779.16 297.0367 5.226 22253.699
33986.5 297.2512 5.1722 22255.199
34086.27 297.2718 4.9672 22255.721
34274.89 297.5085 5.0913 22257.18"""

measurements = []
for line in data.split('\n'):
    parts = line.split()
    measurements.append([float(part) for part in parts])

# Define initial state and covariance
x_hat = np.zeros((9, 1))  # Initial state vector with zeros
P = np.eye(9) * 0.1        # Initial covariance matrix

# Process measurements
estimated_positions = []
times = []
for idx, measurement in enumerate(measurements):
    z = np.array(measurement).reshape(-1, 1)
    x_hat, P = kalman_filter(x_hat, P, z)
    estimated_positions.append((x_hat[0], x_hat[1], x_hat[2]))  # Include elevation
    times.append(measurement[-1])

# Plot estimated target positions versus time
plt.figure(figsize=(10, 5))
plt.plot(times, [pos[0] for pos in estimated_positions], label='Estimated Range (MR)')
plt.plot(times, [pos[1] for pos in estimated_positions], label='Estimated Azimuth (MA)')
plt.plot(times, [pos[2] for pos in estimated_positions], label='Estimated Elevation (ME)')
plt.xlabel('Time')
plt.ylabel('Estimated Value')
plt.title('Estimated Target Positions versus Time')
plt.legend()
plt.grid(True)
plt.show()
