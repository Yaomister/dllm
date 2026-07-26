import math
from .scheduler import Scheduler


class InverseScheduler(Scheduler):
    def get_temperature(self, step: int):
        progress = self.get_progress(step)
        return self.min_temp + (progress) * (self.max_temp - self.min_temp)
