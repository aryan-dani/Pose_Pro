"""
Configuration module for PosePro
Contains all configuration constants, thresholds, and color definitions
"""

import os

# Flask configuration
SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(32))

# Trajectory tracking
TRAJECTORY_BUFFER_SIZE = 150

# Color definitions (BGR format for OpenCV)
COLORS = {
    'primary': (0, 255, 0),
    'secondary': (0, 200, 0),
    'accent': (50, 255, 50),
    'warning': (255, 255, 0),
    'danger': (255, 50, 50),
    'info': (0, 255, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'trajectory_start': (0, 255, 0),
    'trajectory_mid': (255, 255, 0),
    'trajectory_end': (255, 100, 0)
}

# Detection thresholds for rep counting and scoring
THRESHOLDS = {
    'rep_start': 15.0,              # Angle to start rep
    'rep_peak_min': 40.0,           # Minimum angle to be considered peak
    'rep_end': 12.0,                # Angle to end rep
    'ideal_rom_min': 50.0,          # Ideal ROM minimum
    'ideal_rom_max': 90.0,          # Ideal ROM maximum
    'max_symmetry_diff': 20.0,      # Max acceptable symmetry difference
    'max_torso_tilt': 15.0,         # Max acceptable torso tilt
    'ideal_elbow_min': 150.0,       # Ideal elbow angle minimum
    'ideal_elbow_max': 180.0,       # Ideal elbow angle maximum
    'min_rep_duration': 0.3,        # Minimum duration for a valid rep
    'peak_hold_frames': 1,          # Frames to confirm peak detection
    'angle_smoothing_window': 2,    # Window size for angle smoothing
    'hysteresis_margin': 3.0        # Margin to prevent oscillation
}

# Scoring weights for overall score calculation
SCORING_WEIGHTS = {
    'rom': 0.25,
    'symmetry': 0.20,
    'torso_stability': 0.20,
    'smoothness': 0.15,
    'elbow_position': 0.20
}

# Grade thresholds
GRADE_THRESHOLDS = [
    (90, 'A+'),
    (85, 'A'),
    (80, 'A-'),
    (75, 'B+'),
    (70, 'B'),
    (65, 'B-'),
    (60, 'C+'),
    (55, 'C'),
    (0, 'F')
]

# Camera settings
CAMERA_SETTINGS = {
    'default_width': 1280,
    'default_height': 720,
    'mac_width': 640,
    'mac_height': 480,
    'fps': 30,
    'buffer_size': 1,
    'jpeg_quality': 70
}

# MediaPipe settings
MEDIAPIPE_SETTINGS = {
    'pose_complexity_default': 1,
    'pose_complexity_mac': 0,
    'detection_confidence_default': 0.75,
    'detection_confidence_mac': 0.6,
    'tracking_confidence_default': 0.75,
    'tracking_confidence_mac': 0.6,
    'visibility_threshold': 0.3
}

# Server settings
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5005
