import numpy as np
import random
from config import *

class PopulationGenerator:
    def __init__(self, num_patients=NUM_PATIENTS):
        self.num_patients = num_patients

    def generate(self):
        population = []

        for i in range(self.num_patients):
            patient = {
                "id": i,
                "age_group": random.choice(AGE_GROUPS),
                "deprivation": random.choice(DEPRIVATION_LEVELS),
                "appointment_type": random.choice(APPOINTMENT_TYPES),
                "behaviour_type": random.choice(BEHAVIOUR_TYPES),
                "fatigue": 0.0
            }
            population.append(patient)

        return population
