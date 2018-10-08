import numpy as np
from numpy.linalg import inv
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

    def update(self, arm, reward, context):

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
        self.n_dims = n_dims

        # hyperparameter governing exploit/explore tradeoff
        self.alpha = alpha

        # the A matrix and B vectors used in ridge regression for each arm
        self.A = np.zeros(shape=(n_arms, n_dims, n_dims))
        self.B = np.zeros(shape=(n_arms, n_dims))

        # keep track of which actions we've already used
        self.seen = np.full(shape=(n_arms), fill_value=False)

    def choose(self, t, context):
        contexts = np.split(context, self.n_arms)
        Q = [self._estimate_arm_value(i, contexts[i]) for i in range(self.n_arms)]
        return np.argmax(Q)

    def _estimate_arm_value(self, arm, context):

        # if arm has not been seen
        if not self.seen[arm]:
            self.A[arm] = np.identity(self.n_dims)
            self.B[arm] = np.zeros(shape=(self.n_dims))
            self.seen[arm] = True

        # generate params matrix
        theta = inv(self.A[arm]).dot(self.B[arm])

        # run ridge regression
        p = theta.dot(context) + self.alpha*np.sqrt( context.dot(inv(self.A[arm]).dot(context)) )

        return p

    def update(self, arm, reward, context):
        context = np.split(context, self.n_arms)[arm]
        self.A[arm] += np.outer(context, context)
        self.B[arm] += reward*context


def read_data():
    data = np.loadtxt("melb/dataset.txt")
    arms = data[:,0]
    rewards = data[:,1]
    contexts = data[:, 2:]
    return arms, rewards, contexts


def off_policy_train(mab, arms, rewards, contexts, T):

    n_arms = np.max(arms)
    history = []

    for j in range(len(arms)):
        
        arm = int(arms[j])
        reward = rewards[j]
        context = contexts[j]

        if arm == mab.choose(j, context):
            mab.update(arm, reward, context)
            history.append(reward)

    return history


if __name__ == "__main__":

    arms, rewards, contexts = read_data()

    #mab = EpsilonGreedyBandit(10, 0.05)
    mab = LinUCB(10, 10, 0.80)
    history = off_policy_train(mab, arms, rewards, contexts, None)
    #print(history)
    print(np.mean(history))

    """
    env = SimpleEnvironment(3, [100, 5.8, 5.5], [0.5, 0.5, 0.5])
    bandit = EpsilonGreedyBandit(3, 0.15)

    arms, r, contexts = read_data()
    print(arms.shape)
    print(r.shape)
    print(contexts.shape)

    for i in range(1000):
        #arm = bandit.choose()
        bandit.update(arm, env.step(arm))
        print(bandit.Q)

    """
