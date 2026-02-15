from population import PopulationGenerator
from environment import NHSEnvironment
from dqn import DQNAgent
from plots import plot_robustness_results, save_results_excel

import numpy as np
import random
import torch


# =========================
# Reproducibility
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# Fairness Metrics
# =========================

def fairness_metrics(env, agent, episodes=300):

    group_rewards = {
        "low": [],
        "medium": [],
        "high": []
    }

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(episodes):

        state = env.reset()
        done = False
        deprivation = env.current_patient["deprivation"]
        episode_reward = 0

        while not done:
            action_index = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_index)
            episode_reward += reward
            state = next_state

        group_rewards[deprivation].append(episode_reward)

    agent.epsilon = original_epsilon

    avg_rewards = {k: np.mean(v) for k, v in group_rewards.items()}
    gap = max(avg_rewards.values()) - min(avg_rewards.values())

    overall = np.mean(
        group_rewards["low"]
        + group_rewards["medium"]
        + group_rewards["high"]
    )

    return overall, gap


# =========================
# Training
# =========================

def train_agent(fairness_penalty, seed=0):

    set_seed(seed)

    population = PopulationGenerator().generate()
    env = NHSEnvironment(population, fairness_penalty=fairness_penalty)

    agent = DQNAgent(state_dim=14, action_dim=5, epsilon=1.0)

    num_episodes = 1000
    epsilon_min = 0.05
    epsilon_decay = 0.995

    for episode in range(num_episodes):

        state = env.reset()
        done = False

        while not done:
            action_index = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_index)

            agent.replay_buffer.push(
                state,
                action_index,
                reward,
                next_state,
                done
            )

            agent.train_step()
            state = next_state

        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

    return agent


# =========================
# Robustness Comparison
# =========================

def robustness_comparison(agent_standard, agent_fair):

    print("\n=== ROBUSTNESS COMPARISON ===\n")

    population = PopulationGenerator().generate()

    scenarios = {
        "Base": {},
        "Attendance Drop": {"attendance_shift": -0.15},
        "Budget Cut": {"budget_scale": 0.5},
        "Effectiveness Drop": {"effectiveness_scale": 0.7}
    }

    results = {}

    for name, params in scenarios.items():

        print(f"\nScenario: {name}")

        # Standard policy
        env_std = NHSEnvironment(
            population,
            fairness_penalty=0.0,
            **params
        )

        overall_std, gap_std = fairness_metrics(env_std, agent_standard)

        # Fairness-aware policy
        env_fair = NHSEnvironment(
            population,
            fairness_penalty=1.0,
            **params
        )

        overall_fair, gap_fair = fairness_metrics(env_fair, agent_fair)

        print("Standard Policy:")
        print(f"  Reward: {overall_std:.2f}")
        print(f"  Gap: {gap_std:.2f}")

        print("Fairness-Aware Policy:")
        print(f"  Reward: {overall_fair:.2f}")
        print(f"  Gap: {gap_fair:.2f}")

        results[name] = {
            "standard_reward": overall_std,
            "fair_reward": overall_fair,
            "standard_gap": gap_std,
            "fair_gap": gap_fair
        }

        print("-" * 50)

    # 🔥 Generate PNG plots automatically
    plot_robustness_results(results)
    save_results_excel(results)

# =========================
# Main
# =========================

def main():

    print("\nTraining Standard Agent (penalty=0.0)...\n")
    agent_standard = train_agent(fairness_penalty=0.0)

    print("\nTraining Fairness-Aware Agent (penalty=1.0)...\n")
    agent_fair = train_agent(fairness_penalty=1.0)

    robustness_comparison(agent_standard, agent_fair)


if __name__ == "__main__":
    main()
