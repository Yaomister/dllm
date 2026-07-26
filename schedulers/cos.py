import math
from .scheduler import Scheduler

class CosScheduler(Scheduler):
    def get_temperature(self, step: int):
        progress = self.get_progress(step)
        return self.min_temp + (self.max_temp - self.min_temp) * 0.5 * (1 + math.cos(math.pi * progress))
