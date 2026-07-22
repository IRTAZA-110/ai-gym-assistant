# exercise_logic.py
import mediapipe as mp
import numpy as np

# Define landmark constants for easy access
mp_pose = mp.solutions.pose
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value
RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value
LEFT_ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value
RIGHT_ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value

# A simple dictionary to hold state information that needs to persist across calls for a specific exercise.
# This is a cleaner approach than using global variables for everything.
exercise_state_globals = {
    'bicep_curl_start_elbow_pos': None
}


def bicep_curl_counter(detector, stage, rep_counter):
    """
    Implements the logic for counting Bicep Curls with form correction.
    - Tracks left arm by default.
    - Checks for excessive elbow movement during the curl.
    """
    angle = detector.calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    feedback = ""
    feedback_type = "info"

    if angle is not None:
        if angle > 160:
            # Arm is extended, this is our 'down' state.
            # Store the elbow's starting position when the arm is fully extended.
            stage = "down"
            if detector.lm_list:
                exercise_state_globals['bicep_curl_start_elbow_pos'] = detector.lm_list[LEFT_ELBOW][1:]  # Store [x, y]
            feedback = "Arm extended"

        # Check if user has curled up and the previous stage was 'down'
        if angle < 30 and stage == 'down':
            # --- Form Correction Logic ---
            start_pos = exercise_state_globals.get('bicep_curl_start_elbow_pos')
            if start_pos and detector.lm_list:
                current_elbow_pos = np.array(detector.lm_list[LEFT_ELBOW][1:])
                start_elbow_pos = np.array(start_pos)

                # Calculate the distance the elbow has moved from its starting position
                elbow_movement_distance = np.linalg.norm(current_elbow_pos - start_elbow_pos)

                # If elbow moves more than a threshold (e.g., 40 pixels), provide corrective feedback
                if elbow_movement_distance > 40:
                    feedback = "Keep Your Elbow Still!"
                    feedback_type = "warning"  # A new type for bad form
                else:
                    # If form is good, count the rep
                    stage = "up"
                    rep_counter += 1
                    feedback = "Rep Complete!"
                    feedback_type = "success"
            else:
                # Fallback if starting position wasn't captured
                stage = "up"
                rep_counter += 1
                feedback = "Rep Complete!"
                feedback_type = "success"

    return stage, rep_counter, feedback, feedback_type


def shoulder_press_counter(detector, stage, rep_counter):
    """Logic for counting Shoulder Press (uses right arm)."""
    angle = detector.calculate_angle(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW)
    feedback = ""
    feedback_type = "info"

    if angle is not None:
        if angle < 90:  # Arm is down
            stage = "down"
            feedback = "Press Up!"
        if angle > 160 and stage == 'down':  # Arm is extended up
            stage = "up"
            rep_counter += 1
            feedback = "Great Press!"
            feedback_type = "success"

    return stage, rep_counter, feedback, feedback_type


def side_raise_counter(detector, stage, rep_counter):
    """Logic for counting Side Raises (uses left arm)."""
    angle = detector.calculate_angle(LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW)
    feedback = ""
    feedback_type = "info"

    if angle is not None:
        if angle < 20:  # Arm is down
            stage = "down"
            feedback = "Raise arm to the side."
        if angle > 80 and stage == 'down':  # Arm is at T-pose
            stage = "up"
            rep_counter += 1
            feedback = "Excellent Raise!"
            feedback_type = "success"

    return stage, rep_counter, feedback, feedback_type


def overhead_clap_counter(detector, stage, rep_counter):
    """Logic for counting Overhead Claps."""
    distance = detector.calculate_distance(LEFT_WRIST, RIGHT_WRIST)
    feedback = "Raise hands and clap!"
    feedback_type = "info"

    if distance is not None and detector.lm_list:
        left_wrist_y = detector.lm_list[LEFT_WRIST][2]
        left_shoulder_y = detector.lm_list[LEFT_SHOULDER][2]

        if left_wrist_y < left_shoulder_y:  # Hands are above shoulders
            if distance > 150:  # Hands are apart
                stage = "apart"
                feedback = "Clap above head!"
            if distance < 50 and stage == 'apart':  # Hands are close (clap)
                stage = "clap"
                rep_counter += 1
                feedback = "Clap!"
                feedback_type = "success"
        else:
            stage = "down"  # Stage when hands are not high enough
            feedback = "Raise hands higher!"

    return stage, rep_counter, feedback, feedback_type


def jumping_jack_counter(detector, stage, rep_counter):
    """Logic for counting Jumping Jacks."""
    feedback = "Jump!"
    feedback_type = "info"

    left_arm_angle = detector.calculate_angle(LEFT_HIP, LEFT_SHOULDER, LEFT_WRIST)
    right_arm_angle = detector.calculate_angle(RIGHT_HIP, RIGHT_SHOULDER, RIGHT_WRIST)

    if left_arm_angle is not None and right_arm_angle is not None:
        # Check if arms are "in"
        if left_arm_angle < 45 and right_arm_angle < 45:
            stage = "in"

        # Check if arms are "out" and the previous stage was "in"
        if left_arm_angle > 90 and right_arm_angle > 90 and stage == 'in':
            stage = "out"
            rep_counter += 1
            feedback = "Good Jump!"
            feedback_type = "success"

    return stage, rep_counter, feedback, feedback_type