from abc import ABC, abstractmethod

def Scheduler(ABC):

    def __init__(self, max_temp, min_temp, total_steps):
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.total_steps = total_steps

    @abstractmethod
    def get_temperature(self, step: int) -> float:
        ...

    def get_progress(self, step:int) -> float:
        return step / max(self.total_steps - 1, 1)

