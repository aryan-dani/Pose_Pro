"""
Pose Analysis module for PosePro
Contains core pose detection, angle calculations, and rep scoring logic
"""

import math
import numpy as np
import datetime
from collections import deque
from .config import THRESHOLDS, SCORING_WEIGHTS, GRADE_THRESHOLDS


def calculate_shoulder_abduction_from_vertical(shoulder, elbow):
    """
    Calculate shoulder abduction angle from vertical.
    
    Args:
        shoulder: Landmark point with x, y coordinates
        elbow: Landmark point with x, y coordinates
    
    Returns:
        float: Angle in degrees (0° = arms down, 90° = arms horizontal)
    """
    try:
        if not all(hasattr(p, 'x') and hasattr(p, 'y') for p in [shoulder, elbow]):
            return 0.0
            
        shoulder_pt = np.array([shoulder.x, shoulder.y])
        elbow_pt = np.array([elbow.x, elbow.y])

        arm_vector = elbow_pt - shoulder_pt
        # Vertical vector pointing down (in image coordinates where y increases downward)
        vertical_vector = np.array([0, 1])

        dot_product = np.dot(arm_vector, vertical_vector)
        magnitude_arm = np.linalg.norm(arm_vector)

        if magnitude_arm == 0:
            return 0

        cos_angle = dot_product / magnitude_arm
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Error in shoulder abduction calculation: {e}")
        return 0.0


def calculate_torso_tilt_from_vertical(left_shoulder, right_shoulder, left_hip, right_hip):
    """
    Calculate torso tilt angle from true vertical.
    
    Args:
        left_shoulder, right_shoulder: Shoulder landmark points
        left_hip, right_hip: Hip landmark points
    
    Returns:
        float: Absolute tilt angle in degrees
    """
    try:
        if not all(hasattr(p, 'x') and hasattr(p, 'y') for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return 0.0
            
        shoulder_mid = np.array([
            (left_shoulder.x + right_shoulder.x) / 2, 
            (left_shoulder.y + right_shoulder.y) / 2
        ])
        hip_mid = np.array([
            (left_hip.x + right_hip.x) / 2, 
            (left_hip.y + right_hip.y) / 2
        ])

        torso_vector = shoulder_mid - hip_mid
        vertical_vector = np.array([0, -1])

        dot_product = np.dot(torso_vector, vertical_vector)
        magnitude_torso = np.linalg.norm(torso_vector)

        if magnitude_torso == 0:
            return 0

        cos_angle = dot_product / magnitude_torso
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return abs(math.degrees(angle_rad))
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Error in torso tilt calculation: {e}")
        return 0.0


def calculate_elbow_extension_angle(shoulder, elbow, wrist):
    """
    Calculate elbow extension angle (angle at the elbow joint).
    
    Args:
        shoulder, elbow, wrist: Landmark points
    
    Returns:
        float: Angle in degrees (180° = fully extended)
    """
    try:
        a = np.array([shoulder.x, shoulder.y])
        b = np.array([elbow.x, elbow.y])
        c = np.array([wrist.x, wrist.y])
        
        ba = a - b
        bc = c - b
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 170.0
        
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Error in elbow angle calculation: {e}")
        return 170.0


def calculate_rep_scores(rep_data):
    """
    Calculate all component scores and overall score for a rep.
    
    Args:
        rep_data: RepData object containing rep metrics
    
    Modifies rep_data in place with calculated scores
    """
    try:
        # ROM Score
        if THRESHOLDS['ideal_rom_min'] <= rep_data.rom_total <= THRESHOLDS['ideal_rom_max']:
            rep_data.score_rom = 1.0
        elif rep_data.rom_total < THRESHOLDS['ideal_rom_min']:
            rep_data.score_rom = rep_data.rom_total / THRESHOLDS['ideal_rom_min']
        else:
            rep_data.score_rom = max(0.6, 1.0 - ((rep_data.rom_total - THRESHOLDS['ideal_rom_max']) / 20))
        
        # Symmetry Score
        if rep_data.symmetry_diff <= 5:
            rep_data.score_symmetry = 1.0
        else:
            rep_data.score_symmetry = max(0.3, 1.0 - (rep_data.symmetry_diff / 20))
        
        # Torso Stability Score
        if rep_data.torso_lean_max <= 5:
            rep_data.score_torso_stability = 1.0
        else:
            rep_data.score_torso_stability = max(0.2, 1.0 - (rep_data.torso_lean_max / 15))
        
        # Smoothness Score
        if rep_data.avg_angles:
            velocity = np.diff(rep_data.avg_angles)
            if len(velocity) > 1:
                acceleration = np.diff(velocity)
                accel_var = np.var(acceleration) if len(acceleration) > 0 else 0
                rep_data.score_smoothness = max(0.3, 1 - (accel_var / 50))
            else:
                rep_data.score_smoothness = 0.7
        
        # Elbow Position Score
        elbow_avg = rep_data.elbow_angle_avg
        if THRESHOLDS['ideal_elbow_min'] <= elbow_avg <= THRESHOLDS['ideal_elbow_max']:
            rep_data.score_elbow_position = 1.0
        else:
            diff = abs(elbow_avg - 170)  # 170 is ideal
            rep_data.score_elbow_position = max(0.4, 1.0 - (diff / 30))
        
        # Overall Score (weighted average)
        scores = [
            rep_data.score_rom,
            rep_data.score_symmetry,
            rep_data.score_torso_stability,
            rep_data.score_smoothness,
            rep_data.score_elbow_position
        ]
        weights = [
            SCORING_WEIGHTS['rom'],
            SCORING_WEIGHTS['symmetry'],
            SCORING_WEIGHTS['torso_stability'],
            SCORING_WEIGHTS['smoothness'],
            SCORING_WEIGHTS['elbow_position']
        ]
        
        rep_data.score_overall = sum(s * w for s, w in zip(scores, weights)) * 100
        
        # Assign grade
        rep_data.form_grade = get_grade(rep_data.score_overall)
        
        # Generate warnings
        rep_data.warnings = []
        if rep_data.score_symmetry < 0.7:
            rep_data.warnings.append("bilateral_asymmetry")
        if rep_data.score_torso_stability < 0.7:
            rep_data.warnings.append("torso_instability")
        if rep_data.rom_total < 60:
            rep_data.warnings.append("insufficient_rom")
        if rep_data.rom_total > 100:
            rep_data.warnings.append("excessive_rom")
        if rep_data.elbow_angle_avg < 150:
            rep_data.warnings.append("elbow_bending")
        
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Error calculating scores: {e}")
        rep_data.score_overall = 50.0
        rep_data.form_grade = 'C'


def get_grade(score):
    """
    Convert numerical score to letter grade.
    
    Args:
        score: Numerical score (0-100)
    
    Returns:
        str: Letter grade
    """
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return 'F'


class AngleSmoother:
    """Handles angle smoothing for noise reduction in measurements."""
    
    def __init__(self, window_size=None):
        if window_size is None:
            window_size = THRESHOLDS['angle_smoothing_window']
        self.buffer = deque(maxlen=window_size)
    
    def get_smoothed(self, new_angle):
        """Apply moving average smoothing to reduce noise."""
        self.buffer.append(new_angle)
        if len(self.buffer) < 2:
            return new_angle
        return float(np.mean(self.buffer))
    
    def get_velocity(self):
        """Calculate the rate of change of angle."""
        if len(self.buffer) < 2:
            return 0.0
        return self.buffer[-1] - self.buffer[-2]
    
    def clear(self):
        """Clear the smoothing buffer."""
        self.buffer.clear()
