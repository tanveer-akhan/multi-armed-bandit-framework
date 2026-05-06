import matplotlib.pyplot as plt
import numpy as np
import src.utils.util as util


def put_label(ax, title, xlabel=None, ylabel=None, plot=False):
    """
    Sets labels, titles, legend
    Args:
        title : title for graph
        xlabel : label for x axis
        ylabel : label for y axis
    """

    if xlabel is None:
        xlabel = "Steps"
    ax.set_xlabel(xlabel)

    if ylabel is None:
        ylabel = "Rewards"
    ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.legend()

    if plot is True:
        plt.show()

# GOOD


def plot_single_arm_estimate(agent, arm_no, max_steps=None, ax=None, fig=None, custom_label=None, custom_title=None, plot=False):
    """
    Plot graph for one specific arm
    Args :
    agent : the agent whose arms are to be plotted
    arm_no : the arm to be plotted
    max_steps : the maximum no. of steps to be plotted
    ax : axis to plot the graph on | creates new if None is given
    custom_label : Custom label 
    """

    if ax is None:
        fig, ax = plt.subplots()
    if max_steps is None:
        max_steps = agent.epochs_taken
    if custom_title is None:
        custom_title = f"Reward over steps for arm : {arm_no}"

    # Create label
    if custom_label == None:
        label = f"Arm {arm_no}"
    else:
        label = custom_label + f" Arm {arm_no}"

    # Plot estimate
    ax.plot(agent.estimate_history[arm_no]
            [:max_steps],  label=label)

    # Plot true value
    ax.axhline(y=agent.environment.base_truth[arm_no],
               linestyle='--', label=f"True Arm {arm_no}", alpha=0.6)

    put_label(ax, custom_title, plot=plot)

    return fig


def plot_multiple_arm_estimate(agent, ax=None, fig=None, only_top=False, top_count=3, max_steps=None, custom_label=None, custom_title=None, plot=False):
    """
    Plots estimated rewards of multiple arms of an agent
    Args :
    agent : the agent whose arms are to be plotted
    ax : axis to plot the graph on | creates new if None is given
    only_top : if only the top arms should be plotted | Def : All arms to be plotted
    top_count : No. of top estimated arms to be plotted | Def : 3 | 
    max_steps : the maximum no. of steps to be plotted
    custom_label : Custom label | Def : "Graph of multiple arms"
    """

    if ax == None:
        fig, ax = plt.subplots()
    if max_steps is None:
        max_steps = agent.epochs_taken

    incr_arms = np.argsort(agent.estimated_rewards)

    # Top arm and Bottom arm indexes
    top_arm_index = incr_arms[:top_count]
    bot_arm_index = incr_arms[top_count:]

    # Plot bottom arms
    if only_top == False:
        for arm in bot_arm_index:
            plot_single_arm_estimate(
                agent, arm_no=arm, max_steps=max_steps, ax=ax, custom_label=custom_label)
    # Plot top arms
    for arm in top_arm_index:
        plot_single_arm_estimate(
            agent, arm_no=arm, max_steps=max_steps, ax=ax, custom_label=custom_label)

    if custom_label is None:
        custom_title = "Graph of multiple arms"

    put_label(ax, title=custom_title, plot=plot)

    return fig

# LABEL ISSUE


def plot_agent_reward_history(agent, title=None, label=None, ax=None, fig=None, xlabel=None, ylabel=None, plot=False):
    if ax is None:
        fig, ax = plt.subplots()

    if title is None:
        title = "Agent's Entire Reward History."

    if label is None:
        label = f'{agent.strategy.name} : {agent.strategy.value}'

    ax.plot(agent.reward_history, label=label)
    put_label(ax, title=title, xlabel=xlabel, ylabel=ylabel, plot=plot)

    return fig


def plot_comparison(agents, only_top=False, top_count=3, max_steps=None, custom_title=None, plot=True):
    """
    Plot graph for one specific arm
    Args :
    agents : list of agents who are being compared
    only_top : if only the top arms should be plotted | Def : All arms to be plotted
    top_count : No. of top estimated arms to be plotted | Def : 3 | 
    max_steps : the maximum no. of steps to be plotted
    custom_title : Custom title | Def : strategy1  VS  Strategy"
    """

    fig, ax = plt.subplots()

    if max_steps is None:
        max_steps = agents[0].epochs_taken
    if custom_title is None:
        custom_title = f"{agents[0].strategy.name} VS {agents[1].strategy.name} Strategy"

    for agent in agents:
        plot_multiple_arm_estimate(agent, ax, only_top=only_top, top_count=top_count,
                                   max_steps=max_steps, custom_label=agent.strategy.name, custom_title=custom_title, plot=plot)

    put_label(ax, title=custom_title)

    return fig


def compare_reward_history(agent_list, title="Cumulative reward over steps",
                           xlabel="Steps", ylabel="Total Reward", plot=False):

    fig, ax = plt.subplots()

    for current_agent in agent_list:
        label = f"{current_agent.strategy.name} : {current_agent.strategy.value}"

        plot_agent_reward_history(
            current_agent.reward_history, title=title, label=label, ax=ax, fig=fig, xlabel=xlabel, ylabel=ylabel, plot=False)

    put_label(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, plot=plot)

    return fig


def compare_optimal_action(agent_list, title="Optimal Action over steps steps",
                           xlabel="Steps", ylabel="Optimal action taken", plot=False, save_img=False, img_name=None):

    fig, ax = plt.subplots()

    for current_agent in agent_list:
        label = f" {current_agent.strategy.name} : {current_agent.strategy.value} "

        plot_optimal_actions(current_agent, ax=ax, fig=fig, title=title, xlabel=xlabel,
                             ylabel=ylabel, plot=False)

    put_label(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, plot=plot)

    return fig


def plot_optimal_actions(agent, ax=None, fig=None, title=None, xlabel=None, ylabel=None, plot=False):

    if ax is None:
        fig, ax = plt.subplots()

    if title is None:
        title = "Optimal Action Rate"

    ax.plot(agent.optimal_rate, label=agent.strategy.name)

    put_label(ax=ax, title=title,
              xlabel=xlabel, ylabel=ylabel, plot=plot)

    return fig


def plot_cumulative_regret(agent, ax=None, title=None, xlabel=None, ylabel=None, plot=False):
    if ax is None:
        fig, ax = plt.subplots()

    if title is None:
        title = "Regret Curve"

    if ylabel is None:
        ylabel = "Cumulative Regret"

    ax.plot(agent.cumulative_regret, label=agent.strategy.name)
    put_label(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, plot=plot)
    return fig


def plot_regret(agent, ax=None, title=None, xlabel=None, ylabel=None, plot=False):
    if ax is None:
        fig, ax = plt.subplots()

    if title is None:
        title = "Regret Curve"

    if ylabel is None:
        ylabel = "Current Regret"

    label = f'{agent.strategy.name} : {agent.strategy.value}'

    ax.plot(agent.current_regret, label=label)

    put_label(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, plot=plot)
    return fig
