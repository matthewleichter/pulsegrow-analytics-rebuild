
import json
from datetime import datetime

def log_performance(metrics: dict, filepath: str):
    log = {"timestamp": datetime.now().isoformat(), **metrics}
    with open(filepath, "a") as f:
        f.write(json.dumps(log) + "\n")
