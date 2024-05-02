import numpy as np
import matplotlib.pyplot as plt

# Kalman Filter Parameters
F = np.eye(4)  # State transition matrix (identity matrix for linear system)
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # Measurement matrix
Q = np.eye(4) * 0.01  # Process noise covariance matrix
R = np.eye(4) * 0.1  # Measurement noise covariance matrix

def kalman_filter(x_hat, P, z):
    # Prediction step
    x_hat_minus = np.dot(F, x_hat)
    P_minus = np.dot(np.dot(F, P), F.T) + Q
    
    # Measurement update step
    K = np.dot(np.dot(P_minus, H.T), np.linalg.inv(np.dot(np.dot(H, P_minus), H.T) + R))
    x_hat = x_hat_minus + np.dot(K, (z - np.dot(H, x_hat_minus)))
    P = np.dot((np.eye(4) - np.dot(K, H)), P_minus)
    
    return x_hat, P

def multivariate_gaussian(x, mean, covariance):
    n = x.shape[0]
    det = np.linalg.det(covariance)
    inv_covariance = np.linalg.inv(covariance)
    exponent = -0.5 * np.dot(np.dot((x - mean).T, inv_covariance), (x - mean))
    return (1.0 / np.sqrt((2 * np.pi) ** n * det)) * np.exp(exponent)

# Define sensor measurements (range, azimuth, elevation, time)
sensor_measurements = [
    [[100, 30, 20, 1], [110, 35, 25, 1]],  # Sensor 1 measurements
    [[120, 40, 30, 1], [130, 45, 35, 1]]   # Sensor 2 measurements
]

# Define targets
num_targets = 2
target_states = [np.array([[0], [0], [0], [0]]) for _ in range(num_targets)]  # Initial state for each target
target_covariances = [np.eye(4) * 0.1 for _ in range(num_targets)]  # Initial covariance for each target

# Plotting
plt.figure(figsize=(10, 5))
colors = ['b', 'r']

# Joint Probabilistic Data Association (JPDA)
for sensor_idx, measurements in enumerate(sensor_measurements):
    # Compute association probabilities for each target
    association_probabilities = []
    for target_idx, target_state in enumerate(target_states):
        # Compute measurement likelihood for each sensor's measurement
        likelihood = 1.0
        for sensor_measurement in measurements:
            z = np.array(sensor_measurement).reshape(-1, 1)
            predicted_measurement = np.dot(H, target_state)
            innovation = z - predicted_measurement
            innovation_covariance = np.dot(np.dot(H, target_covariances[target_idx]), H.T) + R
            likelihood *= multivariate_gaussian(innovation, np.zeros((4, 1)), innovation_covariance)
        association_probabilities.append(likelihood)
    
    # Normalize association probabilities
    association_probabilities = np.array(association_probabilities)
    association_probabilities /= np.sum(association_probabilities)
    
    # Find most likely associated target
    most_likely_target_idx = np.argmax(association_probabilities)
    most_likely_target_state = target_states[most_likely_target_idx]
    print(f"Sensor {sensor_idx + 1} - Most likely associated target: {most_likely_target_idx + 1}")
    
    # Plot sensor measurements
    for i, (range_, azimuth, elevation, time) in enumerate(measurements):
        plt.scatter(range_, azimuth, color=colors[sensor_idx], label=f"Sensor {sensor_idx + 1} Measurement {i + 1}")

    # Plot estimated target positions
    plt.scatter(most_likely_target_state[0], most_likely_target_state[1], marker='x', color=colors[sensor_idx], label=f"Estimated Target {most_likely_target_idx + 1}")

# Plot settings
plt.xlabel('Range')
plt.ylabel('Azimuth')
plt.title('Sensor Measurements and Estimated Target Positions')
plt.legend()
plt.grid(True)
plt.show()
