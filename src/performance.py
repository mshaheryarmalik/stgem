import time

class PerformanceData:

    def __init__(self):
        self.timers = {}
        self.histories = {}

    def get_history(self, id):
        if not id in self.timers:
            raise Exception("No history for the identifier '{}'.".format(id))
        return self.histories[id]

    def save_history(self, id, value, single=False):
        if not single:
            if not id in self.timers:
                self.histories[id] = []
            self.histories[id].append(value)
        else:
            self.histories[id] = value

    def timer_start(self, id):
        # TODO: Implement a good time for all platforms.
        if id in self.timers and self.timers[id] is not None:
            raise Exception("Restarting timer '{}' without resetting.".format(id))

        self.timers[id] = time.monotonic()

    def timer_reset(self, id):
        if not id in self.timers:
            raise Exception("No timer '{}' to be reset.".format(id))
        if self.timers[id] is None:
            raise Exception("Timer '{}' already reset.".format(id))

        time_elapsed = time.monotonic() - self.timers[id]
        self.timers[id] = None

        return time_elapsed

    def timers_hold(self):
        for id, t in self.timers.items():
            if t is not None:
                self.timers[id] = time.monotonic() - self.timers[id]

    def timers_resume(self):
        self.timers_hold()
