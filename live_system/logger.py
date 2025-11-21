from datetime import datetime
import os

class AccidentLogger:
    """
    Logs accident events to logs/accidents.log
    Each line: timestamp - ACCIDENT on Camera ID
    """

    def __init__(self, log_file="logs/accidents.log"):
        self.log_file = log_file

        # Ensure logs folder exists
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def log(self, camera_id):
        """Write a log entry with timestamp + camera info."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{now} - ACCIDENT on {camera_id}\n"

        with open(self.log_file, "a") as f:
            f.write(entry)

        print(f"[LOGGER] {entry.strip()}")
