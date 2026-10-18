from __future__ import annotations

import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceLandmarkDetector:
    def __init__(
        self,
        model_path: str,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = model_path
        self.num_faces = num_faces
        self.min_face_detection_confidence = min_face_detection_confidence
        self.min_face_presence_confidence = min_face_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.detector = self._create_detector()

        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        self.NOSE_IDX = 1

        # Landmarks for head pose
        self.POSE_IDXS = {
            "nose_tip": 1,
            "chin": 152,
            "left_eye_corner": 33,
            "right_eye_corner": 263,
            "left_mouth": 61,
            "right_mouth": 291,
        }

    def _create_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.num_faces,
            min_face_detection_confidence=self.min_face_detection_confidence,
            min_face_presence_confidence=self.min_face_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        return vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr, timestamp_ms: int):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    @staticmethod
    def _normalized_to_pixel(landmark, width, height):
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return (x, y)

    @staticmethod
    def _euclidean(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_eye_points(self, face_landmarks, frame_shape):
        h, w = frame_shape[:2]

        left_eye = [
            self._normalized_to_pixel(face_landmarks[idx], w, h)
            for idx in self.LEFT_EYE_IDX
        ]
        right_eye = [
            self._normalized_to_pixel(face_landmarks[idx], w, h)
            for idx in self.RIGHT_EYE_IDX
        ]

        return left_eye, right_eye

    def get_nose_position(self, face_landmarks, frame_shape):
        h, w = frame_shape[:2]
        nose_landmark = face_landmarks[self.NOSE_IDX]
        x = int(nose_landmark.x * w)
        y = int(nose_landmark.y * h)
        return (x, y)

    def compute_eye_aspect_ratio(self, eye_points):
        if len(eye_points) != 6:
            return 0.0

        p1, p2, p3, p4, p5, p6 = eye_points

        vertical_1 = self._euclidean(p2, p6)
        vertical_2 = self._euclidean(p3, p5)
        horizontal = self._euclidean(p1, p4)

        if horizontal == 0:
            return 0.0

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def draw_eye_points(self, frame_bgr, eye_points, color=(0, 255, 0)):
        for pt in eye_points:
            cv2.circle(frame_bgr, pt, 2, color, -1)

    def estimate_head_pose(self, face_landmarks, frame_shape):
        h, w = frame_shape[:2]

        image_points = np.array([
            self._normalized_to_pixel(face_landmarks[self.POSE_IDXS["nose_tip"]], w, h),
            self._normalized_to_pixel(face_landmarks[self.POSE_IDXS["chin"]], w, h),
            self._normalized_to_pixel(face_landmarks[self.POSE_IDXS["left_eye_corner"]], w, h),
            self._normalized_to_pixel(face_landmarks[self.POSE_IDXS["right_eye_corner"]], w, h),
            self._normalized_to_pixel(face_landmarks[self.POSE_IDXS["left_mouth"]], w, h),
            self._normalized_to_pixel(face_landmarks[self.POSE_IDXS["right_mouth"]], w, h),
        ], dtype=np.float64)

        model_points = np.array([
            (0.0, 0.0, 0.0),           # nose tip
            (0.0, -63.6, -12.5),       # chin
            (-43.3, 32.7, -26.0),      # left eye corner
            (43.3, 32.7, -26.0),       # right eye corner
            (-28.9, -28.9, -24.1),     # left mouth
            (28.9, -28.9, -24.1),      # right mouth
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = float(euler_angles[0])
        yaw = float(euler_angles[1])
        roll = float(euler_angles[2])

        # Normalize wrapped Euler angles to a usable range
        if pitch < -90:
            pitch = -(180 + pitch)
        elif pitch > 90:
            pitch = 180 - pitch

        if yaw < -90:
            yaw = -(180 + yaw)
        elif yaw > 90:
            yaw = 180 - yaw

        if roll < -90:
            roll = -(180 + roll)
        elif roll > 90:
            roll = 180 - roll

        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
        }

    def process_face(self, frame_bgr, detection_result):
        output = frame_bgr.copy()

        if not detection_result.face_landmarks:
            return output, 0, 0.0, 0.0, 0.0, None, None

        face_landmarks = detection_result.face_landmarks[0]

        left_eye_points, right_eye_points = self.get_eye_points(face_landmarks, output.shape)

        self.draw_eye_points(output, left_eye_points, (0, 255, 0))
        self.draw_eye_points(output, right_eye_points, (0, 255, 0))

        left_ear = self.compute_eye_aspect_ratio(left_eye_points)
        right_ear = self.compute_eye_aspect_ratio(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0

        nose_point = self.get_nose_position(face_landmarks, output.shape)
        cv2.circle(output, nose_point, 4, (0, 0, 255), -1)

        head_pose = self.estimate_head_pose(face_landmarks, output.shape)

        return output, 1, left_ear, right_ear, avg_ear, nose_point, head_pose

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()