import numpy as np
import config


class NHSEnvironment:

    def __init__(
        self,
        population,
        fairness_penalty=0.0,
        attendance_shift=0.0,
        effectiveness_scale=1.0,
        budget_scale=1.0
    ):
        self.population = population
        self.current_patient = None
        self.current_step = 0

        self.fairness_penalty = fairness_penalty
        self.attendance_shift = attendance_shift
        self.effectiveness_scale = effectiveness_scale
        self.budget_scale = budget_scale

        self.reset_budget()

    def reset(self):
        self.current_patient = np.random.choice(self.population)
        self.current_patient["fatigue"] = 0.0
        self.current_step = 0
        self.reset_budget()
        return self._get_state()

    def step(self, action_index):

        action = config.ACTIONS[action_index]
        action = self._apply_intervention(action)

        attendance, reward = self._simulate_attendance(action)

        self.current_step += 1
        done = self.current_step >= config.MAX_APPOINTMENTS_PER_PATIENT

        next_state = self._get_state()

        return next_state, reward, done, {}

    def reset_budget(self):
        self.budget = {
            k: int(v * self.budget_scale)
            for k, v in config.DAILY_BUDGET.items()
        }

    def _apply_intervention(self, action):
        if action == "none":
            return action

        if self.budget[action] <= 0:
            return "none"

        self.budget[action] -= 1
        return action

    def _simulate_attendance(self, action):

        base_prob = (
            config.BASE_ATTENDANCE[
                self.current_patient["behaviour_type"]
            ]
            + self.attendance_shift
        )

        boost = (
            config.INTERVENTION_EFFECTS[action]
            * self.effectiveness_scale
        )

        fatigue_penalty = self.current_patient["fatigue"]

        probability = base_prob + boost - fatigue_penalty
        probability = np.clip(probability, 0.0, 1.0)

        attendance = np.random.rand() < probability

        # Fatigue update
        if action != "none":
            self.current_patient["fatigue"] += config.FATIGUE_DECAY

        self.current_patient["fatigue"] = np.clip(
            self.current_patient["fatigue"], 0.0, 0.5
        )

        # Reward
        if attendance:
            reward = 1.0
        else:
            reward = -2.0

        reward -= config.INTERVENTION_COSTS[action]

        # Fairness penalty
        if (
            self.current_patient["deprivation"] == "high"
            and action == "none"
        ):
            reward -= self.fairness_penalty

        return attendance, reward

    def _get_state(self):

        patient = self.current_patient

        age_encoding = [0, 0, 0]
        age_encoding[
            ["young", "middle", "elderly"].index(patient["age_group"])
        ] = 1

        dep_encoding = [0, 0, 0]
        dep_encoding[
            ["low", "medium", "high"].index(patient["deprivation"])
        ] = 1

        appt_encoding = [0, 0, 0]
        appt_encoding[
            ["gp", "specialist", "mental_health"].index(
                patient["appointment_type"]
            )
        ] = 1

        state = [
            patient["fatigue"],
            self.budget["sms"] / 100,
            self.budget["personal_sms"] / 100,
            self.budget["call"] / 30,
            self.budget["escalate"] / 5,
            *age_encoding,
            *dep_encoding,
            *appt_encoding
        ]

        return np.array(state, dtype=np.float32)
