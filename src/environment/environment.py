import numpy as np
from src.logger_utils.runtime_logs import record_logs


class Environment:

    """
    Creates an object with base_truth with mean for each arm

    Attributes : 
        arm_count        : no. of arms
        max_mean         : maximum mean value of an arm
        base_truth       : np array with means for each arm
        optimal_arm      : the index of the arm with the largest value
    """

    def __init__(self, arm_count: int, max_mean: float, s_d=1):

        self.arm_count = arm_count
        self.max_mean = max_mean
        self.s_d = s_d
        self.base_truth = np.random.uniform(
            0, self.max_mean, size=self.arm_count
        )

        self.optimal_arm = np.argmax(self.base_truth, axis=0)
        record_logs("Environment Created.")

    def get_reward(self, arm):
        arm_mean = self.base_truth[arm]
        reward = np.random.normal(loc=arm_mean, scale=self.s_d)
        return reward

    def reinitialize_env(self, max_mean, s_d=1):
        self.max_mean = max_mean
        self.s_d = s_d
        self.base_truth = np.random.uniform(
            0, self.max_mean, size=self.arm_count
        )

        self.optimal_arm = np.argmax(self.base_truth, axis=0)
        record_logs("Environment Created.")
