import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# 1. Define the input space (x-axis)
X_plot = np.linspace(-5, 5, 500).reshape(-1, 1)

# 2. Define a Kernel (Squared Exponential / RBF kernel)
# length_scale=1.0 controls how smooth/wide the squiggles are
kernel = 1.0 * RBF(length_scale=0.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0)

# -------------------------------------------------------------
# VISUALIZING THE PRIOR
# -------------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Draw 5 random function samples from the prior
y_prior_samples = gp.sample_y(X_plot, n_samples=5, random_state=None)
plt.plot(X_plot, y_prior_samples)
plt.title("Samples from the GP Prior\n(Before seeing data)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)

# -------------------------------------------------------------
# VISUALIZING THE POSTERIOR (After observing data)
# -------------------------------------------------------------
# Define some arbitrary training data points (Observations)
X_train = np.array([[-4], [-1.5], [0], [2], [4]])
y_train = np.array([[-2], [1.5], [0], [1], [-1]])

# Fit the GP to the data
gp.fit(X_train, y_train)

# Sample 5 trajectories using the marginals after data fitting
ys = gp.sample_y(X_plot, n_samples=5)
# Predict the mean and standard deviation across the plot range
y_mean, y_std = gp.predict(X_plot, return_std=True)

plt.subplot(1, 2, 2)
# Plot training data
plt.scatter(X_train, y_train, color='red', zorder=5, label='Observations')
for i in range (5):
    plt.plot(X_plot, ys.T[i])
# Plot predictive mean
plt.plot(X_plot, y_mean, color='blue', label='Predictive Mean')
# Plot uncertainty band (95% confidence interval / 2 standard deviations)
plt.fill_between(X_plot.flatten(), 
                 y_mean - 2 * y_std, 
                 y_mean + 2 * y_std, 
                 color='blue', alpha=0.2, label='95% Confidence Interval')

plt.title("GP Posterior\n(After fitting to observations)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()