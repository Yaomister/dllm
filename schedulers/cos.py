import math
from .scheduler import Scheduler

# The cosine scheduler.
# Using the cosine annealing method from "Stochastic Gradient Descent with Warm Restarts".
# This ensures the mean T_pos is the same as linear when integrated.
class CosScheduler(Scheduler):
    def get_temperature(self, step: int):
        progress = self.get_progress(step)
        return self.min_temp +  0.5 * (self.max_temp - self.min_temp) * (1 + math.cos(math.pi * progress))
