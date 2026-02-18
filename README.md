# 🤖 Fairness-Aware Reinforcement Learning Under Distribution Shift

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Reinforcement Learning-FF6F00?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Deep Q Network-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Power BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black" />
</p>

> Investigating the trade-off between **efficiency and fairness** in reinforcement learning under real-world operational shocks.

---

## 📌 Overview

This project simulates a **public healthcare appointment scheduling system** to evaluate how fairness-aware reward regularization impacts RL agent performance under distribution shift scenarios such as:

- 📉 Attendance drops
- 💰 Budget cuts
- 📊 Intervention effectiveness decay

Two agents are trained and compared:
| Agent | Strategy |
|-------|----------|
| **Standard Policy** | Maximizes total reward only |
| **Fairness-Aware Policy** | Adds disparity penalty across socioeconomic groups |

---

## 🎯 Key Research Question

> *Can we design RL agents that balance efficiency and fairness under distribution shift?*

---

## 🏗️ Environment Design

### 👥 Population
Each synthetic patient is characterized by:
- Age group
- Deprivation level (low / medium / high)
- Appointment type
- Behavioral type & fatigue level

**Population size:** 20,000 synthetic agents (configurable)

### 💊 Interventions (Actions)
| Action | Description |
|--------|-------------|
| None | No intervention |
| SMS | Basic reminder |
| Personalized SMS | Targeted reminder |
| Call | Direct outreach |
| Escalation | High-priority follow-up |

### 📐 Attendance Model
```
P(attend) = base_behavior + intervention_effect - fatigue
```

---

## 🤖 RL Approach — Deep Q-Network (DQN)

- ✅ Experience replay buffer
- ✅ Target network stabilization
- ✅ Epsilon-greedy exploration
- ✅ Multi-episode training

---

## ⚖️ Fairness Metric
```
Reward Gap = Max(Group Reward) - Min(Group Reward)
```
Lower gap = more equitable outcomes across deprivation groups.

---

## 🔬 Distribution Shift Experiments

| Scenario | Description |
|----------|-------------|
| **Base** | Nominal operating conditions |
| **Attendance Drop** | System-wide reduction in attendance probability |
| **Budget Cut** | Reduced intervention capacity |
| **Effectiveness Drop** | Intervention effectiveness scaled down |

---

## 📊 Key Findings

- 🏆 Standard policy achieves slightly higher **base reward**
- ✅ Fairness-aware policy **reduces disparity** under attendance shocks
- ✅ Fairness-aware policy **improves equity** under budget cuts
- ⚠️ Efficiency-fairness trade-offs are **scenario-dependent**
- 📌 Fairness regularization does **not universally dominate**

> This highlights the importance of evaluating RL systems under **distribution shift** rather than only in-distribution performance.

---

## 📈 Visualizations

Plots generated automatically:
- Efficiency performance comparison
- Policy Reward Under Distribution Shift
- Equity gap comparison
- Fairness Gap Under Distribution Shift
- 📊 Interactive **Power BI Dashboard** for scenario-level outcomes

---

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Run the Project
```bash
python main.py
```

This will:
1. Train both agents
2. Run robustness comparison
3. Generate plots
4. Export Excel file for dashboard

### Output Files
```
visual/   → plots and charts
data/     → exported Excel files
```

---

## 🛠️ Technical Highlights

- Custom RL environment design
- Fairness-aware reward shaping
- Robustness evaluation under distribution shift
- Structured experimental comparison
- Dashboard-based analytical reporting

---

## ⚠️ Limitations & Future Work

**Current Limitations:**
- Synthetic environment (no real-world dataset)
- Simplified attendance model
- Fairness defined via reward disparity only

**Future Directions:**
- Multi-objective RL
- Constrained RL optimization
- Causal modeling of attendance
- Real-world healthcare dataset validation
- Distributionally robust RL approaches

---

## 📝 Disclaimer

This project uses **synthetic data** for research and demonstration purposes only. It does not represent real patient data or official healthcare modeling.

---

<p align="center">Made with ❤️ by <a href="https://github.com/ArpitaRandive">Arpita Randive</a></p>
