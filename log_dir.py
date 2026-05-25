"""
log_dir.py — creates a fresh timestamped folder for this run's logs.

Importing this module once sets up logs/run_YYYY-MM-DD_HH-MM-SS/.
Other modules call open_log() to write into that folder. The log file
is automatically named after the calling script — e.g. world_building.py
writes to world_building.jsonl.

Example:
    import log_dir
    f = log_dir.open_log()         # → world_building.jsonl
    f.write("hello\\n")
"""

import os
import sys
import time
from datetime import datetime

# Create the per-run directory once, at import time.
RUN_DIR = os.path.join("logs", f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
os.makedirs(RUN_DIR, exist_ok=True)
print(f"[log_dir] Run logs → {RUN_DIR}")

# Shared start time for all logs/timestamps across the whole project.
# Other modules should use `log_dir.START_TIME` instead of their own time.time()
# captured at import — that way every script reports the same elapsed seconds.
START_TIME = time.time()

def elapsed():
    """Seconds since the run started. Use this everywhere instead of subtracting
    a per-module _start_time so all logs share one time origin."""
    return time.time() - START_TIME

def open_log(extension="jsonl", mode="w", buffering=1):
    """Open a log file inside this run's folder, named after the calling script.
    e.g. called from world_building.py → opens world_building.jsonl.
    `extension` has no leading dot. `buffering=1` is line-buffered."""
    caller_file = sys._getframe(1).f_globals.get("__file__")
    if caller_file is None:
        caller_name = "unknown"
    else:
        caller_name = os.path.splitext(os.path.basename(caller_file))[0]
    filename = f"{caller_name}.{extension}"
    return open(os.path.join(RUN_DIR, filename), mode, buffering=buffering)

def log_path(name):
    """Return the full path for a log file inside this run's folder,
    without opening it (use this if a 3rd-party API wants a path string)."""
    return os.path.join(RUN_DIR, name)
