from .base import Strategy
from .exploit import Exploit
from .explore import Explore

import numpy as np


class EpsilonGreedy(Strategy):

    """
    Epsilon Greedy Class 
    Explores with epsilon probability
    Exploits with 1-epsilon probability
    """

    def __init__(self, epsilon_value: float):
        self.name = "Epsilon Greedy"
        self.value = epsilon_value
        self.explore = Explore()
        self.exploit = Exploit()

    def select_arm(self, agent):  # TODO:Simplify
        val = np.random.uniform(0, 1)
        if (val <= self.value):
            random_arm = self.explore.select_arm(agent)
            return random_arm            
        else:
            arm = self.exploit.select_arm(agent)
            return arm.item()
