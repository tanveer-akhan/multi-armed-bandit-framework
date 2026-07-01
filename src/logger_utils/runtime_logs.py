import os
import datetime


# LOG File generated once per script run
LOG_FILE = "logs/" + "record_" + \
    datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".txt"

os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def take_logs(agent):
    filename = "results/" + "record_" + \
        datetime.now().strftime("%Y%m%d%H%M%S")+".txt"
    with open(filename, "w") as f:
        f.write(str(agent.actions_taken))
        f.write("")
        f.write(str(agent.reward_history))


def record_logs(log: str, filename=None):
    # Write log
    with open(LOG_FILE, "a") as f:
        f.write(log)
        f.write("\n")

    return filename
