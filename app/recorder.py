import os
import time
from collections import deque
import cv2


class AlertClipRecorder:
    def __init__(
        self,
        output_dir: str = "recordings",
        fps: int = 20,
        pre_seconds: int = 3,
        post_seconds: int = 3,
    ):
        self.output_dir = output_dir
        self.fps = fps
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds

        os.makedirs(self.output_dir, exist_ok=True)

        self.buffer = deque(maxlen=self.fps * self.pre_seconds)
        self.writer = None
        self.is_recording = False
        self.post_frames_remaining = 0
        self.current_filepath = None

    def _create_writer(self, frame_shape):
        h, w = frame_shape[:2]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.current_filepath = os.path.join(
            self.output_dir, f"drowsiness_alert_{timestamp}.avi"
        )

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(self.current_filepath, fourcc, self.fps, (w, h))

    def update_buffer(self, frame):
        self.buffer.append(frame.copy())

    def start_event(self, frame):
        if self.is_recording:
            return

        self._create_writer(frame.shape)

        for buffered_frame in self.buffer:
            self.writer.write(buffered_frame)

        self.writer.write(frame)
        self.is_recording = True
        self.post_frames_remaining = 0

    def write_frame(self, frame):
        if self.is_recording and self.writer is not None:
            self.writer.write(frame)

    def stop_event(self):
        if self.is_recording:
            self.post_frames_remaining = self.fps * self.post_seconds

    def tick_post_event(self, frame):
        if not self.is_recording:
            return

        if self.post_frames_remaining > 0:
            self.writer.write(frame)
            self.post_frames_remaining -= 1

        if self.post_frames_remaining == 0 and self.writer is not None:
            self.writer.release()
            self.writer = None
            self.is_recording = False
            self.current_filepath = None

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.is_recording = False