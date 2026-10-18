import csv
import os
import time
from datetime import datetime


class EventLogger:
    def __init__(
        self,
        log_dir: str = "log",
        filename: str = "events.csv",
        cooldown_seconds: float = 5.0,
    ):
        self.log_dir = log_dir
        self.filename = filename
        self.filepath = os.path.join(self.log_dir, self.filename)
        self.cooldown_seconds = cooldown_seconds
        self.last_event_times = {}

        os.makedirs(self.log_dir, exist_ok=True)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "event_type",
                    "avg_ear",
                    "closed_frames",
                    "blink_count",
                    "head_nods",
                    "head_down",
                    "nose_drop",
                    "extra",
                ])

    def _now_str(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def can_log(self, event_type: str) -> bool:
        now = time.time()
        last_time = self.last_event_times.get(event_type, 0.0)

        if now - last_time >= self.cooldown_seconds:
            self.last_event_times[event_type] = now
            return True

        return False

    def log_event(
        self,
        event_type: str,
        avg_ear: float = 0.0,
        closed_frames: int = 0,
        blink_count: int = 0,
        head_nods: int = 0,
        head_down: bool = False,
        nose_drop: float = 0.0,
        extra: str = "",
        use_cooldown: bool = True,
    ):
        if use_cooldown and not self.can_log(event_type):
            return

        with open(self.filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self._now_str(),
                event_type,
                round(avg_ear, 4),
                closed_frames,
                blink_count,
                head_nods,
                int(head_down),
                round(nose_drop, 2),
                extra,
            ])