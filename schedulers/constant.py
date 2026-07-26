import math
from .scheduler import Scheduler

class ConstantScheduler(Scheduler):
    def get_temperature(self, step: int):
        return self.max_temp