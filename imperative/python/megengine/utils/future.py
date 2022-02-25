import threading


class Future:
    def __init__(self, ack=True):
        self.ready = threading.Event()
        self.ack = threading.Event() if ack else None

    def set(self, value):
        self.value = value
        self.ready.set()
        if self.ack:
            self.ack.wait()

    def get(self):
        self.ready.wait()
        if self.ack:
            self.ack.set()
        return self.value
