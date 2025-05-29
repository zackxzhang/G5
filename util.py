import time


class Timer:

    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        self.end = time.perf_counter()
        print('-' * 64)
        print(self.message + ' takes time: ' + str(self.end - self.start))
        print('-' * 64)
