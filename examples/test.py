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
env = environment.Environment(arm_count=15, max_mean=200, s_d=5)

# Create Agent
agent1 = agents.Agent(environment=env, strategy=strategy.EpsilonGreedy(0.2))

# Create Agent
agent2 = agents.Agent(environment=env, strategy=strategy.EpsilonGreedy(0.5))

# Create Agent
agent3 = agents.Agent(environment=env, strategy=strategy.UCB(5))

util.run_experiment(agent=agent1, steps=1000)
util.run_experiment(agent=agent2, steps=1000)
util.run_experiment(agent=agent3, steps=1000)

graph.compare_optimal_action(agent_list=[agent1, agent2, agent3], plot=True)

new_env = environment.Environment(arm_count=15, max_mean=100, s_d=5)


agent1.update_environment(new_environment=new_env)
agent2.update_environment(new_environment=new_env)
agent3.update_environment(new_environment=new_env)


util.run_experiment(agent=agent1, steps=5000)
util.run_experiment(agent=agent2, steps=5000)
util.run_experiment(agent=agent3, steps=5000)

graph.compare_optimal_action(agent_list=[agent1, agent2, agent3], plot=True)
