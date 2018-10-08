import numpy as np
from abc import ABC, abstractmethod


class MultiArmedBandit(ABC):
    """
    Abstract parent to the multi-armed bandit implementation classes.
    """
    @abstractmethod
    def choose_arm(self):
        pass

    @abstractmethod
    def update(r, self):
        pass


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
        
