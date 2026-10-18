import os
import time
from collections import deque

import cv2
import numpy as np

from app.camera import Camera
from app.landmark_detector import FaceLandmarkDetector
from app.alert_system import AlertSystem
from app.logger import EventLogger
from app.recorder import AlertClipRecorder

from config import (
    MODEL_PATH,
    ALARM_PATH,
    LOG_DIR,
    RECORDING_DIR,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    EAR_THRESHOLD,
    CLOSED_FRAMES_THRESHOLD,
    PERCLOS_WINDOW_SIZE,
    PERCLOS_ON_THRESHOLD,
    PERCLOS_OFF_THRESHOLD,
    PITCH_DOWN_THRESHOLD,
    PITCH_RESET_THRESHOLD,
    ROLL_LIMIT,
    YAW_LIMIT,
    NOSE_DROP_THRESHOLD,
    NOSE_RESET_THRESHOLD,
    NOSE_SMOOTHING,
    HEAD_DOWN_FRAME_LIMIT,
)

def draw_panel(panel, title, lines, start_y=40):
    cv2.putText(
        panel, title, (20, start_y),    
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
    )
    y = start_y + 40
    for text, color in lines:
        cv2.putText(
            panel, text, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
        y += 34


def main():
    
    model_path = MODEL_PATH
    sound_path = ALARM_PATH

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Please place the Face Landmarker model in the models folder."
        )

    camera = Camera(camera_index=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    detector = FaceLandmarkDetector(model_path=model_path, num_faces=1)
    alert_system = AlertSystem(sound_path=sound_path, volume=0.8)
    logger = EventLogger(log_dir=LOG_DIR, filename="events.csv", cooldown_seconds=5.0)
    recorder = AlertClipRecorder(output_dir=RECORDING_DIR, fps=20, pre_seconds=3, post_seconds=3)
    
    # State variables
    baseline_frames = 0
    head_down_frames = 0
    closed_frame_count = 0
    blink_count = 0

    drowsy = False
    prev_drowsy = False

    baseline_nose_y = None
    smoothed_nose_y = None
    head_nod_count = 0
    head_down_status = False
    nose_drop = 0.0

    current_pitch = 0.0
    current_yaw = 0.0
    current_roll = 0.0

    pitch_down_condition = False
    nose_down_condition = False

    # PERCLOS state
    perclos_window = deque(maxlen=PERCLOS_WINDOW_SIZE)
    perclos_value = 0.0
    max_perclos = 0.0

    # Session stats
    session_start_time = time.time()
    max_closed_frames = 0
    max_nose_drop = 0.0
    max_pitch_down = 0.0
    drowsy_event_count = 0
    total_drowsy_frames = 0

    camera.open()
    prev_time = time.time()

    logger.log_event(
        event_type="SESSION_START",
        extra="Driver monitoring session started",
        use_cooldown=False,
    )

    try:
        while True:
            frame = camera.read()
            timestamp_ms = int(time.time() * 1000)

            result = detector.detect(frame, timestamp_ms)
            vis_frame, face_count, left_ear, right_ear, avg_ear, nose_point, head_pose = detector.process_face(
                frame, result
            )

            recorder.update_buffer(vis_frame)

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            
            # Blink / Eye closure logic
            if face_count > 0:
                if avg_ear < EAR_THRESHOLD:
                    closed_frame_count += 1
                else:
                    if closed_frame_count > 2:
                        blink_count += 1
                    closed_frame_count = 0
            else:
                closed_frame_count = 0

            max_closed_frames = max(max_closed_frames, closed_frame_count)

            # PERCLOS logic     
            eyes_closed_flag = 1 if avg_ear < EAR_THRESHOLD else 0
            perclos_window.append(eyes_closed_flag)

            if len(perclos_window) > 0:
                perclos_value = sum(perclos_window) / len(perclos_window)
            else:
                perclos_value = 0.0

            max_perclos = max(max_perclos, perclos_value)

            
            # Head pose logic   
            if head_pose is not None:
                current_pitch = head_pose["pitch"]
                current_yaw = head_pose["yaw"]
                current_roll = head_pose["roll"]

                if current_pitch < max_pitch_down:
                    max_pitch_down = current_pitch
            else:
                current_pitch = 0.0
                current_yaw = 0.0
                current_roll = 0.0

            
            # Nose-drop logic   
            if nose_point is not None:
                nose_y = nose_point[1]

                if smoothed_nose_y is None:
                    smoothed_nose_y = float(nose_y)
                else:
                    smoothed_nose_y = (
                        NOSE_SMOOTHING * smoothed_nose_y
                        + (1.0 - NOSE_SMOOTHING) * nose_y
                    )

                if baseline_nose_y is None:
                    baseline_nose_y = smoothed_nose_y
                    baseline_frames = 1
                elif baseline_frames < 60:
                    baseline_nose_y = 0.95 * baseline_nose_y + 0.05 * smoothed_nose_y
                    baseline_frames += 1

                nose_drop = smoothed_nose_y - baseline_nose_y
                max_nose_drop = max(max_nose_drop, nose_drop)
            else:
                smoothed_nose_y = None
                baseline_nose_y = None
                baseline_frames = 0
                nose_drop = 0.0

            
            # Final stable head-down logic
            pitch_down_condition = (
                current_pitch < PITCH_DOWN_THRESHOLD
                and abs(current_roll) < ROLL_LIMIT
                and abs(current_yaw) < YAW_LIMIT
            )

            nose_down_condition = (
                nose_drop > NOSE_DROP_THRESHOLD
                and abs(current_roll) < 25
                and abs(current_yaw) < 30
            )

            head_down_condition = (
                pitch_down_condition
                or (nose_down_condition and current_pitch < -8)
            )

            head_up_condition = (
                current_pitch > PITCH_RESET_THRESHOLD
                and nose_drop < NOSE_RESET_THRESHOLD
            )

            if head_down_condition:
                head_down_frames += 1

                if head_down_frames >= HEAD_DOWN_FRAME_LIMIT:
                    head_down_status = True

            elif head_up_condition:
                if head_down_status:
                    head_nod_count += 1

                head_down_status = False
                head_down_frames = 0
            # else: hold previous state to avoid flicker
            
            # Final drowsiness decision (with hysteresis)
            perclos_condition_on = perclos_value > PERCLOS_ON_THRESHOLD
            perclos_condition_off = perclos_value < PERCLOS_OFF_THRESHOLD

            if not prev_drowsy:
                if (
                    closed_frame_count > CLOSED_FRAMES_THRESHOLD
                    or head_down_status
                    or perclos_condition_on
                ):
                    drowsy = True
                    total_drowsy_frames += 1
                else:
                    drowsy = False

            else:
                if (
                    closed_frame_count < 5
                    and not head_down_status
                    and perclos_condition_off
                ):
                    drowsy = False
                else:
                    drowsy = True
                    total_drowsy_frames += 1

            
            # Alert sound
            if drowsy:
                alert_system.start_alert()
            else:
                alert_system.stop_alert()


            # Recorder
            
            if drowsy and not prev_drowsy:
                recorder.start_event(vis_frame)
            elif drowsy:
                recorder.write_frame(vis_frame)
            elif not drowsy and prev_drowsy:
                recorder.stop_event()
                recorder.tick_post_event(vis_frame)
            else:
                recorder.tick_post_event(vis_frame)

            
            # Logging
            
            if drowsy and not prev_drowsy:
                drowsy_event_count += 1
                logger.log_event(
                    event_type="DROWSINESS_START",
                    avg_ear=avg_ear,
                    closed_frames=closed_frame_count,
                    blink_count=blink_count,
                    head_nods=head_nod_count,
                    head_down=head_down_status,
                    nose_drop=nose_drop,
                    extra=(
                        f"pitch={current_pitch:.2f},"
                        f"yaw={current_yaw:.2f},"
                        f"roll={current_roll:.2f},"
                        f"perclos={perclos_value:.3f},"
                        f"pitch_down_condition={int(pitch_down_condition)},"
                        f"nose_down_condition={int(nose_down_condition)}"
                    ),
                    use_cooldown=True,
                )

            elif not drowsy and prev_drowsy:
                logger.log_event(
                    event_type="DROWSINESS_END",
                    avg_ear=avg_ear,
                    closed_frames=closed_frame_count,
                    blink_count=blink_count,
                    head_nods=head_nod_count,
                    head_down=head_down_status,
                    nose_drop=nose_drop,
                    extra=(
                        f"pitch={current_pitch:.2f},"
                        f"yaw={current_yaw:.2f},"
                        f"roll={current_roll:.2f},"
                        f"perclos={perclos_value:.3f}"
                    ),
                    use_cooldown=False,
                )

            prev_drowsy = drowsy
            session_duration = int(time.time() - session_start_time)

            # On-frame alert text
            
            if drowsy:
                cv2.putText(
                    vis_frame,
                    "DROWSINESS ALERT!",
                    (210, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )

            
            # Right dashboard panel
            
            h, w = vis_frame.shape[:2]
            panel_width = 360
            panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
            panel[:] = (25, 25, 25)

            draw_panel(panel, "LIVE STATUS", [
                (f"Faces: {face_count}", (0, 255, 255)),
                (f"FPS: {fps:.1f}", (200, 200, 255)),
                (f"Avg EAR: {avg_ear:.3f}", (0, 255, 0)),
                (f"PERCLOS: {perclos_value:.2f}", (255, 255, 0)),
                (f"Closed Frames: {closed_frame_count}", (200, 255, 255)),
                (f"Head Down: {'YES' if head_down_status else 'NO'}",
                 (0, 165, 255) if head_down_status else (180, 180, 180)),
                (f"Blinks: {blink_count}", (255, 255, 255)),
                (f"Head Nods: {head_nod_count}", (255, 200, 0)),
            ], start_y=35)

            draw_panel(panel, "HEAD POSE", [
                (f"Pitch: {current_pitch:.2f}", (255, 255, 255)),
                (f"Yaw:   {current_yaw:.2f}", (255, 255, 255)),
                (f"Roll:  {current_roll:.2f}", (255, 255, 255)),
                (f"Nose Drop: {nose_drop:.1f}", (255, 255, 255)),
                (f"PitchCond: {'YES' if pitch_down_condition else 'NO'}",
                 (0, 255, 0) if pitch_down_condition else (120, 120, 120)),
                (f"NoseCond: {'YES' if nose_down_condition else 'NO'}",
                 (0, 255, 0) if nose_down_condition else (120, 120, 120)),
            ], start_y=300)

            draw_panel(panel, "SESSION", [
                (f"Session(s): {session_duration}", (200, 255, 200)),
                (f"Drowsy Events: {drowsy_event_count}", (200, 255, 200)),
                (f"Max Closed: {max_closed_frames}", (200, 255, 200)),
                (f"Max Nose Drop: {max_nose_drop:.1f}", (200, 255, 200)),
                (f"Max Pitch Down: {max_pitch_down:.1f}", (200, 255, 200)),
                (f"Max PERCLOS: {max_perclos:.2f}", (200, 255, 200)),
                (f"Recorder: {'ON' if recorder.is_recording else 'OFF'}",
                 (0, 255, 0) if recorder.is_recording else (150, 150, 150)),
            ], start_y=520)

            canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
            canvas[:, :w] = vis_frame
            canvas[:, w:] = panel

            cv2.imshow("Driver Drowsiness Detection", canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        session_duration = int(time.time() - session_start_time)

        logger.log_event(
            event_type="SESSION_SUMMARY",
            avg_ear=avg_ear if "avg_ear" in locals() else 0.0,
            closed_frames=max_closed_frames,
            blink_count=blink_count,
            head_nods=head_nod_count,
            head_down=head_down_status,
            nose_drop=max_nose_drop,
            extra=(
                f"duration_sec={session_duration};"
                f"drowsy_events={drowsy_event_count};"
                f"total_drowsy_frames={total_drowsy_frames};"
                f"max_pitch_down={max_pitch_down:.2f};"
                f"max_perclos={max_perclos:.3f}"
            ),
            use_cooldown=False,
        )

        logger.log_event(
            event_type="SESSION_END",
            extra="Driver monitoring session ended",
            use_cooldown=False,
        )

        recorder.close()
        alert_system.stop_alert()
        alert_system.cleanup()
        detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()