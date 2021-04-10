import time

class Profiler:

    def __init__(self):
        self.start = time.time()

    def capture(self):
        return (time.time() - self.start) * 1000