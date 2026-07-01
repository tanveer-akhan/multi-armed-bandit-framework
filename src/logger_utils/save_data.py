import os
import csv
from datetime import datetime

CSV_FILE = "results/records_" + \
    datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".csv"


def log_run(agent, avg_reward, total_regret):
    with open("runs.csv", "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            datetime.now(),
            type(agent.strategy).__name__,
            agent.environment.arm_count,
            getattr(agent.strategy, "value", None),
            avg_reward,
            total_regret
        ])


def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "strategy", "arm",
                            "current reward", "cumulative reward", "optimal"])


def record_csv(epoch, strategy, arm, reward, cumulative, optimal):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, strategy, arm, reward, cumulative, optimal])
