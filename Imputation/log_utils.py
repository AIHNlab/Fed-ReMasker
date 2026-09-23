import datetime
import sys


class Tee:
    """Write to both stdout and a log file simultaneously."""

    def __init__(self, *files):
        self._files = files

    def write(self, obj):
        for f in self._files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self._files:
            f.flush()


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def setup_log_file(path):
    """Redirect stdout to both terminal and a log file."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)
    log(f"Logging to {path}")
