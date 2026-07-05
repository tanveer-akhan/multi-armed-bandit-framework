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
    random_seed = 88
    np.random.seed(random_seed)

    env = Environment(arm_count=10, max_mean=200, s_d=12)

    a = Agent(env, EpsilonGreedy(0.3))
    b = Agent(env, UCB(2))
    # run_experiment(agent=a, steps=10000)
    # run_experiment(agent=b, steps=10000)
    # compare_policies([a, b], 10000)
    average_over_runs(a, 10000, 20)
    # plot_multiple_arm_estimate(agent=a, only_top=True, plot=True)
    # compare_reward_history([a, b], plot=True)
    plot_agent_reward_history(a, plot=True)


if __name__ == "__main__":
    main()
