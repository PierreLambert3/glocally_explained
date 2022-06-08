import threading

class Shared_variable:
    def __init__(self, value = None):
        self.value = value
        self.lock = threading.Lock()

    def delete(self):
        self.value = None
        self.lock  = None

    def get_and_set(self, value):
        ret = None
        with self.lock:
            ret = self.value
            self.value = value
        return ret

    def set(self, value):
        with self.lock:
            self.value = value

    def get(self):
        ret = None
        with self.lock:
            ret = self.value
        return ret
