# Third-part libraries
import sys
import numpy as np
import matplotlib.pyplot as plt

# Core
import src.environment.environment as environment
import src.agent.agents as agents

# Strategy
import src.strategy as strategy

# Utilities
import src.graphs.graph as graph
import src.utils.util as util

# Logging
import src.logger_utils.runtime_logs as runtime_logs

# ========================Code Start==========#

# Create Environment
env = environment.Environment(arm_count=10, max_mean=200, s_d=1)

# Create Agent
agent1 = agents.Agent(environment=env, strategy=strategy.UCB(2))
agent2 = agents.Agent(environment=env, strategy=strategy.EpsilonGreedy(0.3))

all_agents = [agent1, agent2]

util.compare_policies(agent_list=all_agents, steps=1000)
fig = graph.plot_agent_reward_history(agent1, plot=True)

util.save_image_to_disk(fig, "t3")
