import numpy as np
from src.logger_utils.runtime_logs import record_logs
import matplotlib.pyplot as plt
from src.agent.agents import Agent

from rich.progress import Progress
from rich.panel import Panel


def show_info(agent):

    optimal_action_perc = (np.sum(agent.optimal_rate)/agent.epochs_taken)*100

    avg_reward = (agent.cumulative_reward/agent.epochs_taken)

    return (
        f"Epoch no : {agent.epochs_taken:>4} | "

        f"Optimal action % : {optimal_action_perc:>7.3f}% | "

        f"Avg. reward : {avg_reward:>7.3f} per action | "
    )


def run_experiment(agent: Agent, steps: int):
    """
    Trains agent over multiple steps
    Args :
        agent : the agent to be trained
        steps : no. of steps the agent is to be trained.
    """

    with Progress() as progress:
        task = progress.add_task("Running Epochs : ", total=steps)

        for i in range(steps):

            agent.update()

            progress.update(task, advance=1)

            if (i % 100 == 0):
                progress.console.print(Panel(show_info(agent), title="Status"))

        record_logs("Experiment Run Completed.\n")


def save_image_to_disk(fig, filename: str):  # FORCES TO SAVE IN RESULTS
    """
    Saves the plot
    Arg : 
        filename : name of the file
    """

    if filename:
        if not filename.endswith('.png'):
            filename += '.png'

        save_path = "results/" + filename
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print("Image saved")
        plt.close(fig)
        record_logs(f"Image Saved : {save_path}.")


def average_over_runs(agent, steps: int, runs: int):

    avg_cum_reward = []
    avg_current_regret = []
    avg_optimal_rate = []

    with Progress() as progress:

        task = progress.add_task("Running experiments : ", total=runs)

        for _ in range(runs):
            temp_agent = Agent(environment=agent.environment,
                               strategy=agent.strategy)

            run_experiment(agent=temp_agent, steps=steps)

            avg_cum_reward.append(temp_agent.reward_history)
            avg_current_regret.append(temp_agent.current_regret)
            avg_optimal_rate.append(temp_agent.optimal_rate)

            progress.update(task, advance=1)

    agent.current_regret = np.mean(avg_current_regret, axis=0)
    agent.reward_history = np.mean(avg_cum_reward, axis=0)  # TODO:Check sanity
    agent.optimal_rate = np.mean(avg_optimal_rate, axis=0)


def compare_policies(agent_list: Agent, steps: int):
    """
    Returns multiple agents run 
    Args :
        agents : list of agents
        steps : no. of steps the agents is to be run.
    """
    for agent in agent_list:
        run_experiment(agent, steps)
        record_logs(
            f"Run for agent with strategy : {agent.strategy.name} completed.\n")
