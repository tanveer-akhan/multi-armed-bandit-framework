import numpy as np

from src.logger_utils.runtime_logs import record_logs
from src.logger_utils.save_data import init_csv, record_csv, log_run


class Agent():

    """
    Creates a learning agent for the k-Armed Bandit problem.

    The agent interacts with an environment by selecting arms according to
    a chosen strategy, receiving rewards, and updating reward estimates
    over time.

    Features tracked by the agent include:

    - Estimated reward values for each arm
    - Number of times each arm has been selected
    - Reward history
    - Cumulative reward
    - Regret per step and cumulative regret
    - Optimal action selection rate
    - Estimate history for visualization/logging

    Attributes :
        environment : Environment object
        strategy : Strategy object

    """

    def __init__(self, environment, strategy):

        # Attributes taken from environment object
        self.environment = environment

        # Taken from strategy
        self.strategy = strategy

        # New attributes defined
        self.estimated_rewards = np.zeros(
            self.environment.arm_count)  # By default defined as 0

        # self.past_strategy_used = {strategy: 0}
        self.actions_taken = []
        self.epochs_taken = 0
        self.optimal_action_history = []

        # Declared for update_estimate
        self.action_counts = np.zeros(shape=environment.arm_count)
        self.reward_history = []
        self.cumulative_reward = 0
        self.reward_history_per_arm = []

        for _ in range(self.environment.arm_count):
            self.reward_history_per_arm.append([])

        self.estimate_history = []

        # Creating history
        for _ in range(self.environment.arm_count):
            self.estimate_history.append([])

        self.optimal_rate = np.zeros(10)

        # CSV File initiated
        init_csv()

        # Regret calculation
        self.current_regret = []
        self.cumulative_regret = []

        filename = record_logs(
            f"Agent Initialized with strategy : {self.strategy.name}.\n")

    def change_strategy(self, new_strategy):
        self.strategy = new_strategy

    def select_arm(self):
        """
        Returns arm selected based on the strategy
        """
        arm_selected = self.strategy.select_arm(self)
        return arm_selected

    def update_estimate_history(self):
        for i in range(self.environment.arm_count):
            self.estimate_history[i].append(self.estimated_rewards[i])

    # TODO:Remove redundancy in zeros
    def update_reward_history_per_arm(self, arm_selected, reward):
        for i in range(self.environment.arm_count):
            if arm_selected == i:
                self.reward_history_per_arm[i].append(reward)
            else:
                self.reward_history_per_arm[i].append(0)

    def update(self,):
        arm = self.select_arm()
        reward = self.environment.get_reward(arm)

        self.update_estimate(arm_selected=arm, reward=reward)

        self.actions_taken.append(arm)

        self.epochs_taken += 1

        isOptimal = (arm == self.environment.optimal_arm).item()
        self.optimal_action_history.append(isOptimal)

        self.cumulative_reward += reward
        self.reward_history.append(reward)

        self.update_reward_history_per_arm(arm_selected=arm, reward=reward)

        self.update_estimate_history()

        self.optimal_rate = np.cumsum(
            self.optimal_action_history) / np.arange(1, len(self.optimal_action_history)+1)

        step_regret = (
            self.environment.base_truth[self.environment.optimal_arm]) - reward
        self.current_regret.append(step_regret)

        if len(self.cumulative_regret) == 0:
            self.cumulative_regret.append(step_regret)
        else:
            self.cumulative_regret.append(
                self.cumulative_regret[-1] + step_regret
            )

        record_csv(epoch=self.epochs_taken,
                   strategy=self.strategy.name,
                   arm=arm,
                   reward=reward,
                   cumulative=self.cumulative_reward,
                   optimal=isOptimal)

        if (self.epochs_taken % 100 == 0):
            record_logs(
                f"Epoch: {self.epochs_taken} | Arm: {arm} | Current Reward: {reward:.3f} | Cumulative Reward : {self.cumulative_reward:.3f} | Current Strategy : {self.strategy.name}")

        return reward

    def update_estimate(self, arm_selected, reward):
        self.action_counts[arm_selected] += 1  # Track of each arm selected

        old_value = self.estimated_rewards[arm_selected]

        self.estimated_rewards[arm_selected] = old_value + (
            1/self.action_counts[arm_selected])*(reward - old_value)

    def initialize_optimistically(self, value: float, increment: bool = False):
        """
        initialize estimated rewards

        Args : 
                value : initialize to specific value
                increment : initial estimate = base_truth +  'value' 
        """

        if increment:
            self.estimated_rewards = (self.environment.base_truth + value)
        else:
            self.estimated_rewards = np.ones(self.environment.arm_count)*value
        record_logs("Initial Estimates Set to Optimistic estimates.")

    def initialize_zeros(self):
        """
        Initialize estimated rewards to 0

        """
        self.estimated_rewards = np.zeros(self.environment.arm_count)

    def update_environment(self, new_environment):
        self.environment = new_environment
