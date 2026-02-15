import numpy as np

# Population
NUM_PATIENTS = 20000
MAX_APPOINTMENTS_PER_PATIENT = 10


AGE_GROUPS = ["young", "middle", "elderly"]
DEPRIVATION_LEVELS = ["low", "medium", "high"]
APPOINTMENT_TYPES = ["gp", "specialist", "mental_health"]

BEHAVIOUR_TYPES = [
    "reliable",
    "forgetful",
    "resistant",
    "fatigue_sensitive",
    "context_sensitive"
]

# Base attendance probabilities
BASE_ATTENDANCE = {
    "reliable": 0.9,
    "forgetful": 0.7,
    "resistant": 0.5,
    "fatigue_sensitive": 0.75,
    "context_sensitive": 0.65
}

# Intervention effects
INTERVENTION_EFFECTS = {
    "none": 0.0,
    "sms": 0.15,
    "personal_sms": 0.25,
    "call": 0.35,
    "escalate": 0.50
}


# Intervention costs
INTERVENTION_COSTS = {
    "none": 0.0,
    "sms": 0.1,
    "personal_sms": 0.15,
    "call": 0.4,
    "escalate": 0.8
}

# Fatigue decay
FATIGUE_DECAY = 0.02

# Budget per day
DAILY_BUDGET = {
    "sms": 100,
    "personal_sms": 100,
    "call": 30,
    "escalate": 5
}

ACTIONS = ["none", "sms", "personal_sms", "call", "escalate"]
FAIRNESS_PENALTY = 0.5
