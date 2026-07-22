# pose_detector.py

import cv2
import mediapipe as mp
import numpy as np




class PoseDetector:
    def __init__(self, mode=False, complexity=0, smooth=True, detection_con=0.5, track_con=0.5): # <-- Change complexity to 0
        """Initializes the PoseDetector with MediaPipe configurations."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=mode,
            model_complexity=complexity, # <-- This will now use the new default value
            smooth_landmarks=smooth,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        # ... rest of the code
        self.mp_draw = mp.solutions.drawing_utils
        self.lm_list = []
        self.results = None

    def find_pose(self, frame, draw=True):
        """
        Processes a video frame to find and draw pose landmarks.
        Returns the annotated frame.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(image_rgb)

        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(frame, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame

    def find_landmarks(self, frame):
        """
        Extracts the pose landmarks into a list with their pixel coordinates.
        Returns the list of landmarks.
        """
        self.lm_list = []
        if self.results.pose_landmarks:
            h, w, c = frame.shape
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
        return self.lm_list

    def calculate_angle(self, p1_idx, p2_idx, p3_idx):
        """
        Calculates the angle between three landmarks using their indices.
        Returns the angle in degrees, or None if landmarks are not visible.
        """
        if len(self.lm_list) == 0: return None

        try:
            p1 = np.array(self.lm_list[p1_idx][1:])
            p2 = np.array(self.lm_list[p2_idx][1:])
            p3 = np.array(self.lm_list[p3_idx][1:])

            radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
            angle = np.abs(np.degrees(radians))

            if angle > 180.0:
                angle = 360 - angle

            return angle
        except IndexError:
            return None

    def calculate_distance(self, p1_idx, p2_idx):
        """
        Calculates the pixel distance between two landmarks.
        Returns the distance, or None if landmarks are not visible.
        """
        if len(self.lm_list) == 0: return None

        try:
            p1 = np.array(self.lm_list[p1_idx][1:])
            p2 = np.array(self.lm_list[p2_idx][1:])

            distance = np.linalg.norm(p1 - p2)
            return distance
        except IndexError:
            return None