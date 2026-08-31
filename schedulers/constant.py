
from .scheduler import Scheduler

# The constant scheduler.
class ConstantScheduler(Scheduler):
    def get_temperature(self, step: int):
        return self.max_temp