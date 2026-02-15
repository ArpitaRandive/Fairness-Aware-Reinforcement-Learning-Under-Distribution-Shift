import matplotlib.pyplot as plt
import numpy as np
import os
from openpyxl import Workbook



def plot_robustness_results(results_dict):

    os.makedirs("visual", exist_ok=True)

    scenarios = list(results_dict.keys())

    standard_rewards = [results_dict[s]["standard_reward"] for s in scenarios]
    fair_rewards = [results_dict[s]["fair_reward"] for s in scenarios]

    standard_gaps = [results_dict[s]["standard_gap"] for s in scenarios]
    fair_gaps = [results_dict[s]["fair_gap"] for s in scenarios]

    x = np.arange(len(scenarios))

    # Consistent style
    plt.style.use("seaborn-v0_8-whitegrid")

    # =========================
    # Reward Plot
    # =========================
    plt.figure(figsize=(9, 5))

    plt.plot(
        x, standard_rewards,
        marker="o", linewidth=2.5, markersize=8,
        label="Standard Policy"
    )

    plt.plot(
        x, fair_rewards,
        marker="o", linewidth=2.5, markersize=8,
        label="Fairness-Aware Policy"
    )

    plt.xticks(x, scenarios, rotation=20, fontsize=11)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title("Policy Reward Under Distribution Shift", fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()

    reward_path = os.path.join("visual", "robustness_reward.png")
    plt.savefig(reward_path, dpi=300)
    plt.close()

    # =========================
    # Gap Plot
    # =========================
    plt.figure(figsize=(9, 5))

    plt.plot(
        x, standard_gaps,
        marker="o", linewidth=2.5, markersize=8,
        label="Standard Policy"
    )

    plt.plot(
        x, fair_gaps,
        marker="o", linewidth=2.5, markersize=8,
        label="Fairness-Aware Policy"
    )

    plt.xticks(x, scenarios, rotation=20, fontsize=11)
    plt.ylabel("Reward Gap (Max − Min)", fontsize=12)
    plt.title("Fairness Gap Under Distribution Shift", fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()

    gap_path = os.path.join("visual", "robustness_gap.png")
    plt.savefig(gap_path, dpi=300)
    plt.close()

    print("Updated visuals saved in 'visual/' folder.")

def save_results_excel(results_dict):

    os.makedirs("data", exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Robustness Results"

    # Header
    ws.append(["Scenario", "Policy", "Reward", "Gap"])

    for scenario, values in results_dict.items():

        # Standard row
        ws.append([
            scenario,
            "Standard",
            values["standard_reward"],
            values["standard_gap"]
        ])

        # Fairness row
        ws.append([
            scenario,
            "Fairness-Aware",
            values["fair_reward"],
            values["fair_gap"]
        ])

    file_path = os.path.join("data", "robustness_results.xlsx")
    wb.save(file_path)

    print("Saved robustness_results.xlsx in 'data/' folder.")
