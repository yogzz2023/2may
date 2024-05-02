import numpy as np

# Define parameters
num_targets = 3
num_measurements = 3
sensor_range = 100
measurement_variance = 10
target_variance = 5

# Generate random target positions and measurements
np.random.seed(0)
true_targets = sensor_range * np.random.rand(num_targets, 2)  # (x, y)
measurements = true_targets + np.random.normal(0, measurement_variance, (num_targets, 2))

# Initialize JPDA probabilities
prob_association = np.ones((num_targets, num_measurements)) / num_measurements

# Perform JPDA update
for _ in range(10):  # Iterations
    # Predict next state (assuming constant velocity)
    true_targets += np.random.normal(0, target_variance, (num_targets, 2))

    # Calculate predicted measurements
    predicted_measurements = true_targets + np.random.normal(0, measurement_variance, (num_targets, 2))

    # Compute likelihoods
    likelihoods = np.zeros((num_targets, num_measurements))
    for i in range(num_targets):
        for j in range(num_measurements):
            likelihoods[i, j] = np.exp(-0.5 * np.linalg.norm(measurements[j] - predicted_measurements[i]) ** 2 / measurement_variance)

    # Update association probabilities
    prob_association = likelihoods * prob_association
    prob_association /= np.sum(prob_association, axis=1, keepdims=True)

# Estimate target positions
estimated_targets = np.sum(prob_association[:, :, np.newaxis] * true_targets[:, np.newaxis, :], axis=0)

# Print estimated target positions
print("True target positions:\n", true_targets)
print("\nEstimated target positions:\n", estimated_targets)
