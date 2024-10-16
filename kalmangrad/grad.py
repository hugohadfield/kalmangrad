
import numpy as np
import math

from bayesfilter.distributions import Gaussian
from bayesfilter.filtering import BayesianFilter
from bayesfilter.observation import Observation
from bayesfilter.model import StateTransitionModel
from bayesfilter.smoothing import RTS


def transition_func(y, delta_t, n):
    """
    Computes the new state vector new_y at time t + delta_t, given the current state vector y at time t,
    for a Kalman filter of order n.
    The state vector is [y, y', y'', ..., y^(n-1)]^T
    """
    new_y = np.zeros_like(y)
    for i in range(n+1):
        s = 0.0
        for k in range(i, n+1):
            s += y[k] * (delta_t ** (k - i)) / math.factorial(k - i)
        new_y[i] = s
    return new_y


def transition_matrix(delta_t, n):
    """
    Returns the transition matrix for a Kalman filter of order n.
    """
    A = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(n+1):
            if j >= i:
                A[i, j] = (delta_t ** (j - i)) / math.factorial(j - i)
    return A


def observation_func(state):
    """
    Returns the observation vector from the state vector.
    We always observe the first element of the state vector.
    """
    return np.array([state[0]])


def jac_observation_func(state):
    """
    Returns the jacobian of the observation vector from the state vector.
    """
    return np.array([1.0] + [0.0]*(len(state)-1)).reshape(1, len(state))


def grad(
    y: np.ndarray, 
    t: np.ndarray, 
    n: int = 1,
    delta_t = None,
    obs_noise_std = 1e-2
) -> np.ndarray:
    """
    The data y is sampled at times t. The function returns the gradient of y with respect to t. 
    delta_t is the step in t used to run the kalman filter. n is the max order derivative calculated.
    """
    if len(y) != len(t):
        raise ValueError("The length of y and t must be the same.")

    y_array = np.array(y)
    t_array = np.array(t)
    
    # Check the time step
    if delta_t is None:
        delta_t = abs(np.mean(np.diff(t_array))/4.0)
    if delta_t <= 0:
        raise ValueError("delta_t must be positive.")
    if t[-1] - t[0] < 2*delta_t:
        raise ValueError("The time range must be at least 2*delta_t.")
    print('delta_t, ', delta_t)

    # Transition matrix
    jac_mat = transition_matrix(delta_t, n)
    def transition_jacobian_func(state, delta_t):
        return jac_mat

    # Process model
    covariance = 1e-16*np.eye(n+1)
    covariance[n, n] = 1e-4
    transition_model = StateTransitionModel(
        lambda x, dt: transition_func(x, dt, n), 
        covariance,
        transition_jacobian_func
    )

    # Initial state
    initial_state_mean = np.zeros(n+1)
    initial_state_mean[0] = y_array[0]
    initial_state = Gaussian(initial_state_mean, np.eye(n+1))

    # Create observations
    observations = []
    for i in range(len(y_array)):
        new_obs = Observation(
            y_array[i],
            (obs_noise_std**2)*np.eye(1),
            observation_func = observation_func,
            jacobian_func = jac_observation_func
        )
        observations.append(new_obs)

    # Run a bayesian filter
    filter = BayesianFilter(transition_model, initial_state)
    filter_states, filter_times = filter.run(observations, t_array, 1.0/delta_t, use_jacobian=True)
    smoother = RTS(filter)
    smoother_states = smoother.apply(filter_states, filter_times, use_jacobian=False)
    return smoother_states, filter_times


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate noisy sinusoidal data with random time points
    np.random.seed(0)
    t = sorted(np.random.uniform(0.0, 10.0, 100))
    noise_std = 0.01
    y = np.sin(t) + noise_std * np.random.randn(len(t))
    true_first_derivative = np.cos(t)
    true_second_derivative = -np.sin(t)

    # Estimate derivatives using the Kalman filter
    N = 2  # Order of the highest derivative to estimate
    smoother_states, filter_times = grad(y, t, n=N)

    # Extract estimated derivatives
    estimated_position = [state.mean()[0] for state in smoother_states]
    estimated_first_derivative = [state.mean()[1] for state in smoother_states]
    estimated_second_derivative = [state.mean()[2] for state in smoother_states]

    # Plot the results
    plt.figure(figsize=(12, 9))

    # Position
    plt.subplot(3, 1, 1)
    plt.plot(t, y, 'k.', label='Noisy Observations')
    plt.plot(filter_times, estimated_position, 'b-', label='Estimated Position')
    plt.plot(t, np.sin(t), 'r--', label='True Position')
    plt.legend(loc='upper right')
    plt.ylim(-1.5, 1.5)
    plt.title('Position')

    # First Derivative
    plt.subplot(3, 1, 2)
    plt.plot(filter_times, estimated_first_derivative, 'b-', label='Estimated First Derivative')
    plt.plot(t, true_first_derivative, 'r--', label='True First Derivative')
    plt.plot(
        t,
        np.gradient(y, t),
        'k-',
        label='np.gradient calculated derivative'
    )
    plt.legend(loc='upper right')
    plt.ylim(-1.5, 1.5)
    plt.title('First Derivative')

    # Second Derivative
    plt.subplot(3, 1, 3)
    plt.plot(filter_times, estimated_second_derivative, 'b-', label='Estimated Second Derivative')
    plt.plot(t, true_second_derivative, 'r--', label='True Second Derivative')
    plt.legend(loc='upper right')
    plt.ylim(-1.5, 1.5)
    plt.title('Second Derivative')

    plt.tight_layout()
    plt.show()