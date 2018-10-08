import numpy as np

class SimpleEnvironment:
    """
    Test environment for multi-armed bandit.
    """
    def __init__(self, n_arms, mu, sigma):
        self.n_arms = n_arms
        self.mu = mu
        self.sigma = sigma

    def step(self, arm):
        """
        Execute the action associated with the specified arm.
        """
        # sample from the given probability distribution
        reward = np.random.normal(mu[arm], sigma[arm])
        return reward


if __name__ == "__main__":
    n_arms = 3
    mu = [1.2, 0.9, 1.8]
    sigma = [0.5, 0.5, 0.5]
    env = SimpleEnvironment(n_arms, mu, sigma)
    for i in range(10):
        print(env.step(np.random.randint(n_arms)))
