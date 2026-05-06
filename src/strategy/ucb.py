from .base import Strategy

import numpy as np


class UCB(Strategy):

    def __init__(self, c):
        self.name = "UCB"
        self.value = c

    def select_arm(self, agent):
        for i in range(len(agent.action_counts)):
            if agent.action_counts[i] == 0:
                return i
        total_count = np.sum(agent.action_counts)

        explore_criteria = self.value * \
            np.sqrt(np.log(total_count)/(np.array(agent.action_counts)))
        arm = np.argmax(agent.estimated_rewards + explore_criteria)
        return arm
