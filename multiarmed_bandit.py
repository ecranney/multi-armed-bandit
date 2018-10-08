import numpy as np
from np.linalg import inv
from abc import ABC, abstractmethod

from environment import *


class MultiArmedBandit(ABC):
    """
    Abstract parent to the multi-armed bandit implementation classes.
    """
    @abstractmethod
    def choose(self, t, context):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass


class EpsilonGreedyBandit(MultiArmedBandit):
    """
    Implementation of Epsilon Greedy Bandit.
    """
    def __init__(self, n_arms, epsilon):

        # parameters
        self.n_arms = n_arms
        self.epsilon = epsilon

        # current expectations / arm counts
        self.Q = np.full(shape=(n_arms), fill_value=np.inf)
        self.N = np.full(shape=(n_arms), fill_value=0)

        # timestep
        self.t = 0

    def choose(self, t, context):

        # with probability epsilon, explore: choose random arm
        if np.random.rand() < self.epsilon:
            arm = np.random.randint(self.n_arms)

        # else exploit: choose arm with highest q value
        else:
            arm = np.argmax(self.Q)

        return arm

    def update(self, arm, reward):

        # update expected reward for the arm
        # if we have never used this arm before
        if self.N[arm] == 0:
            self.Q[arm] = reward

        # otherwise if we have used this arm before
        else:
            W_history = (self.N[arm] / (self.N[arm]+1))
            W_reward = (1.0 / (self.N[arm]+1))
            self.Q[arm] = self.Q[arm]*W_history + reward*W_reward

        # increment the count of times we've used this arm
        self.N[arm] += 1

        # update the timestep
        self.t = self.t + 1


class LinUCB(MultiArmedBandit):

    def __init__(self, n_arms, n_dims, alpha):

        # number of arms
        self.n_arms = n_arms

        # hyperparameter governing exploit/explore tradeoff
        self.alpha = alpha

        # the A matrix and B vectors used in ridge regression for each arm
        self.A = np.zeros(shape=(n_arms, n_dims, n_dims))
        self.B = np.zeros(shape=(n_arms, 1))

        # keep track of which actions we've already used
        self.seen = np.full(shape=(n_arms), False)

    def choose(self, t, context):
        Q = [_estimate_arm_value(i, context[:,i]) for i in range(self.n_arms)]
        return np.argmax(Q)

    def _estimate_arm_value(arm, context):

        # if arm has not been seen
        if not self.seen[arm]:
            self.A[arm] = np.identity(self.n_dims)
            self.B[arm] = np.zeros(shape=(self.n_dims, 1))

        # generate params matrix
        theta = np.matmul(inv(self.A[arm]), self.B[arm])

        # run ridge regression
        p = np.matmul(theta.T, context) + self.alpha*np.sqrt(\
                np.matmul(context.T, inv(self.A[arm])) )

        return p

    def update(self, arm, reward):
        self.A[arm] += np.matmul(context, context.T)
        self.B[arm] += reward*context


if __name__ == "__main__":
    env = SimpleEnvironment(3, [100, 5.8, 5.5], [0.5, 0.5, 0.5])
    bandit = EpsilonGreedyBandit(3, 0.15)

    for i in range(1000):
        arm = bandit.choose()
        bandit.update(arm, env.step(arm))
        print(bandit.Q)

"""
class MAB(ABC):

    @abstractmethod
    def choose_arm(play, context):
        pass

    @abstractmethod
    def update_expected_rewards(reward, arm, context):
        pass


class EpsilonGreedyBandit(MAB):

    def __init__(self, narms, epsilon):
        pass

    def choose_arm(self, context):
        pass

    def update_expected_rewards(reward, arm, context):
        pass


class UpperConfidenceBoundBandit(MAB):

    def __init__(self, narms, rho):
        
        # number of arms
        self.narms = narms

        # relative importance of exploration
        self.rho = rho

        # average rewards, action execution counts
        self.mu = np.full(shape=(narms), fill_value=np.inf)
        self.n = np.zeros(shape=(narms))

        # current timestep
        self.t = 0

    def choose_arm(self, context):
        
        # choose the action that maximizes q = mu + sqrt(rho*log(t)/n)
        q = self.mu + np.sqrt(self.rho * np.log(self.t) / n )

        # return the argmax
        return np.argmax(q)

    def update_expected_reward(reward, arm, context):
        
        pass
"""
