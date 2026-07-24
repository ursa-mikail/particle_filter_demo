import numpy as np
import matplotlib.pyplot as plt

# Initialize the variables
x = 0.1  # initial actual state
x_N = 1  # Noise covariance in the system (process noise)
x_R = 1  # Noise covariance in the measurement
T = 75   # duration of the chase (number of iterations)
N = 10   # number of particles

# Initialize our initial, prior particle distribution as a Gaussian around the true initial value
V = 2  # variance of the initial estimate
x_P = x + np.sqrt(V) * np.random.randn(N)

# The functions used by the Quail are:
# x = 0.5*x + 25*x/(1 + x**2) + 8*np.cos(1.2*(t-1)) + sqrt(x_N)*np.random.randn()
# z = x**2/20 + sqrt(x_R)*np.random.randn()

# Generate the observations from the randomly selected particles, based on the given function
z_out = [x**2 / 20 + np.sqrt(x_R) * np.random.randn()]
x_out = [x]
x_est = x
x_est_out = [x_est]

for t in range(1, T + 1):
    # Update the flight position and observed position
    x = 0.5 * x + 25 * x / (1 + x**2) + 8 * np.cos(1.2 * (t - 1)) + np.sqrt(x_N) * np.random.randn()
    z = x**2 / 20 + np.sqrt(x_R) * np.random.randn()
    
    # Particle filter
    x_P_update = 0.5 * x_P + 25 * x_P / (1 + x_P**2) + 8 * np.cos(1.2 * (t - 1)) + np.sqrt(x_N) * np.random.randn(N)
    z_update = x_P_update**2 / 20
    
    # Generate the weights for each of these particles
    P_w = (1 / np.sqrt(2 * np.pi * x_R)) * np.exp(-(z - z_update)**2 / (2 * x_R))
    
    # Normalize to form a probability distribution
    P_w /= np.sum(P_w)
    
    # Resampling
    cumulative_sum = np.cumsum(P_w)
    cumulative_sum[-1] = 1.0  # avoid rounding errors
    indexes = np.searchsorted(cumulative_sum, np.random.rand(N))
    
    x_P = x_P_update[indexes]
    
    # The final estimate
    x_est = np.mean(x_P)
    
    # Save data for later plotting
    x_out.append(x)
    z_out.append(z)
    x_est_out.append(x_est)

# Plotting
t = np.arange(T + 1)
plt.figure(figsize=(10, 6))
plt.plot(t, x_out, '.-b', label='True flight position')
plt.plot(t, x_est_out, '-.r', linewidth=3, label='Particle filter estimate')
plt.xlabel('Time step')
plt.ylabel('Quail flight position')
plt.legend()
plt.title('Particle Filter applied to a non-linear model')
plt.grid()
plt.show()