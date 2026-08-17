import math
from .scheduler import Scheduler

# using the cosine annealing method from "Stochastic Gradient Descent with Warm Restarts"
class CosScheduler(Scheduler):
    def get_temperature(self, step: int):
        progress = self.get_progress(step)
        return self.min_temp +  0.5 * (self.max_temp - self.min_temp) * (1 + math.cos(math.pi * progress))
