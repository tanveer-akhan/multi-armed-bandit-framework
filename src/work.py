# Third-part libraries
import sys
import numpy as np
import matplotlib.pyplot as plt


# Modules imported

# Environment
from .environment import Environment

# Agent
from .agent import Agent

# Strategies
from .strategy import Strategy, base, EpsilonGreedy, Exploit, Explore, UCB

# Supporting Utils
from .graphs import compare_optimal_action, compare_reward_history, plot_cumulative_regret, plot_agent_reward_history, plot_comparison, plot_multiple_arm_estimate, plot_optimal_actions, plot_regret
from .utils import run_experiment, save_image_to_disk, average_over_runs, compare_policies
from .logger_utils import take_logs, record_logs, init_csv, record_csv, log_run


def main():
    # Random Seed #TODO: None for now
    random_seed = 88
    np.random.seed(random_seed)

    # Creating Environment
    env = Environment(arm_count=12, max_mean=12.9)

    # Select Agent
    agent = Agent(environment=env, strategy=Explore())

    # Run experiment
    run_experiment(agent=agent, steps=10000)

main()