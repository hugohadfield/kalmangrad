
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
    
    # Check the time step
    if delta_t is None:
        delta_t = abs(np.mean(np.diff(t))/4.0)
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
    initial_state_mean[0] = y[0]
    initial_state = Gaussian(initial_state_mean, np.eye(n+1))

    # Create observations
    observations = []
    for i in range(len(y)):
        new_obs = Observation(
            y[i],
            (obs_noise_std**2)*np.eye(1),
            observation_func = observation_func,
            jacobian_func = jac_observation_func
        )
        observations.append(new_obs)

    # Run a bayesian filter
    filter = BayesianFilter(transition_model, initial_state)
    filter_states, filter_times = filter.run(observations, t, 1.0/delta_t, use_jacobian=True)
    smoother = RTS(filter)
    smoother_states = smoother.apply(filter_states, filter_times, use_jacobian=False)
    return smoother_states, filter_times


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate data
    np.random.seed(0)
    # t = np.linspace(0, 10, 100)
    t = np.random.uniform(0.0, 10.0, 100)
    t.sort()
    noise_std = 0.001
    y = np.sin(t) + noise_std*np.random.randn(len(t))
    ydash = np.cos(t)

    # Run the Kalman filter
    N = 2
    filter_states, filter_times = grad(y, t, n=N)

    # Plot the results
    plt.plot(t, y, label=f"True data {0}")
    plt.plot(t, ydash, label=f"True data {1}")
    plt.plot(filter_times, [fs.mean()[0] for fs in filter_states], label=f"Filtered data {0}")
    plt.plot(filter_times, [fs.mean()[1] for fs in filter_states], label=f"Filtered data {1}")
    plt.plot(t, np.gradient(y, t))
    plt.legend()
    plt.show()