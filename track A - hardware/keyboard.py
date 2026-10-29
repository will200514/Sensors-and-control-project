# keyboard.py - stub to bypass the broken macOS backend in the real package
def is_pressed(*args, **kwargs):
    return False

def read_key(*args, **kwargs):
    return input("key (stub, press enter): ")

def wait(*args, **kwargs):
    input("waiting (stub, press enter to continue)...")

def __getattr__(name):
    def _stub(*args, **kwargs):
        print(f"[keyboard stub] {name}() called, ignored")
    return _stub