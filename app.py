# Suppress warnings - MUST be at the very top before any imports
import os
import sys
import warnings
import logging

# Set environment variables BEFORE importing any ML libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0=all, 1=info, 2=warning, 3=error)
os.environ['GLOG_minloglevel'] = '3'  # Suppress Google logging
os.environ['GRPC_VERBOSITY'] = 'ERROR'  # Suppress gRPC logging
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'  # Suppress abseil logging
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Disable GPU to reduce warnings

# Filter Python warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*inference_feedback_manager.*')
warnings.filterwarnings('ignore', message='.*SymbolDatabase.GetPrototype.*')

# Temporarily redirect stderr during MediaPipe import to suppress C++ warnings
import contextlib

class SuppressStderr:
    """Context manager to suppress stderr output during import."""
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

# Import OpenCV first
import cv2

# Import MediaPipe with stderr suppression
with SuppressStderr():
    import mediapipe as mp

import numpy as np
from collections import deque
import datetime
import math
import json
import uuid
import time
import threading
from flask import Flask, render_template, jsonify, request, Response, session
import base64
from io import BytesIO

# Configure absl logging after import
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
except ImportError:
    pass

# Import database module
from database import (
    init_database, create_session, save_rep, complete_session,
    get_session_history, get_session_reps, check_personal_records,
    get_overall_stats, get_weekly_progress, get_recent_sessions, get_personal_records,
    get_week_stats, get_month_stats, get_score_trend,
    get_user_profile, update_user_profile, create_user,
    get_leaderboard, get_user_rank, get_user_achievements
)

# MediaPipe helpers
mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_pose = mp.solutions.pose  # type: ignore
mp_selfie = mp.solutions.selfie_segmentation  # type: ignore

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(32))

# Initialize database
init_database()

# Global variables
tracking_active = False
camera_active = False
cap = None
frame_counter = 0
current_db_session_id = None  # Track current database session

# Initialize frames with placeholder (will be replaced by camera)
placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(placeholder, "Initializing Camera...", (150, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
current_frame = placeholder.copy()
current_mesh_frame = placeholder.copy()

# Trajectory tracking
trajectory_buffer_size = 150
left_wrist_trajectory = deque(maxlen=trajectory_buffer_size)
right_wrist_trajectory = deque(maxlen=trajectory_buffer_size)
trajectory_colors = []

# Colors
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

# Chat history (simple in-memory storage)
chat_history = []

class RepData:
    def __init__(self):
        self.rep_number = 0
        self.start_time = 0.0
        self.end_time = 0.0
        self.rep_duration = 0.0
        self.left_angles = []
        self.right_angles = []
        self.avg_angles = []
        self.timestamps = []
        self.torso_tilts = []
        self.shoulder_elevations = []
        self.elbow_angles_left = []
        self.elbow_angles_right = []
        self.wrist_positions = []
        self.left_wrist_trajectory = []
        self.right_wrist_trajectory = []
        
        # Metrics
        self.rom_peak_angle = 0.0
        self.rom_min_angle = 0.0
        self.rom_total = 0.0
        self.symmetry_diff = 0.0
        self.torso_lean_max = 0.0
        self.smoothness_index = 0.0
        self.elbow_angle_avg = 0.0
        self.time_between_reps = 0.0
        
        # Scores
        self.score_rom = 0.0
        self.score_symmetry = 0.0
        self.score_torso_stability = 0.0
        self.score_smoothness = 0.0
        self.score_elbow_position = 0.0
        self.score_overall = 0.0
        self.form_grade = "C"
        self.warnings = []

class SessionData:
    def __init__(self):
        self.start_time = datetime.datetime.now()
        self.reps = []
        self.frame_data = []
        self.baseline_hip_y = 0.0
        self.rep_durations = []
        self.last_rep_end_time = 0.0

# Global session data
session_data = SessionData()
current_rep_data = None
rep_state = "idle"  # idle, up_phase, peak_confirmed, down_phase, complete

# Rep tracking state variables
peak_angle_recorded = 0.0          # Track the peak angle reached
peak_confirmation_frames = 0       # Counter for peak confirmation
last_rep_completion_time = 0.0     # Debounce for rep completion
angle_velocity_buffer = deque(maxlen=5)  # Track angle velocity for better phase detection

# Thresholds - Made more lenient for better rep detection
THRESHOLDS = {
    'rep_start': 15.0,            # Angle to start rep (lowered from 25)
    'rep_peak_min': 40.0,         # Minimum angle to be considered peak (lowered from 60)
    'rep_end': 12.0,              # Angle to end rep (lowered from 20)
    'ideal_rom_min': 50.0,        # Ideal ROM minimum (lowered from 70)
    'ideal_rom_max': 90.0,
    'max_symmetry_diff': 20.0,    # Max symmetry difference (increased from 15)
    'max_torso_tilt': 15.0,       # Max torso tilt (increased from 10)
    'ideal_elbow_min': 150.0,     # Ideal elbow angle minimum (lowered from 160)
    'ideal_elbow_max': 180.0,
    'min_rep_duration': 0.3,      # Minimum duration for a valid rep (lowered from 0.5)
    'peak_hold_frames': 1,        # Frames to confirm peak detection (lowered from 2)
    'angle_smoothing_window': 2,  # Window size for angle smoothing (lowered from 3)
    'hysteresis_margin': 3.0      # Margin to prevent oscillation (lowered from 5)
}

# Angle smoothing buffer for noise reduction
angle_buffer = deque(maxlen=THRESHOLDS['angle_smoothing_window'])

def debug_print(message):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

def calculate_shoulder_abduction_from_vertical(shoulder, elbow):
    """Calculate shoulder abduction angle from vertical (0° = arms down, 90° = arms horizontal)"""
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
        
        # angle_deg is the angle from vertical:
        # 0° = arm pointing straight down (at rest)
        # 90° = arm pointing horizontally (lateral raise peak)
        # 180° = arm pointing straight up
        return angle_deg
    except Exception as e:
        debug_print(f"Error in shoulder abduction calculation: {e}")
        return 0.0

def calculate_torso_tilt_from_vertical(left_shoulder, right_shoulder, left_hip, right_hip):
    """Calculate torso tilt angle from true vertical"""
    try:
        if not all(hasattr(p, 'x') and hasattr(p, 'y') for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return 0.0
            
        shoulder_mid = np.array([(left_shoulder.x + right_shoulder.x)/2, 
                               (left_shoulder.y + right_shoulder.y)/2])
        hip_mid = np.array([(left_hip.x + right_hip.x)/2, 
                           (left_hip.y + right_hip.y)/2])

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
        debug_print(f"Error in torso tilt calculation: {e}")
        return 0.0

def calculate_elbow_extension_angle(shoulder, elbow, wrist):
    """Calculate elbow extension angle"""
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
        debug_print(f"Error in elbow angle calculation: {e}")
        return 170.0

def process_frame_data(landmarks, frame_counter, timestamp):
    """Process frame-level data with improved validation"""
    frame_metrics = {}
    
    try:
        if not landmarks or len(landmarks) <= 25:
            return frame_metrics
        
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Check visibility with lenient threshold for key landmarks
        visibility_threshold = 0.3  # Lowered threshold for better detection
        
        # Check all key landmarks have acceptable visibility (only check shoulders)
        if any(lm.visibility < visibility_threshold for lm in [left_shoulder, right_shoulder]):
            return frame_metrics
        
        # Calculate metrics
        left_abduction = calculate_shoulder_abduction_from_vertical(left_shoulder, left_elbow)
        right_abduction = calculate_shoulder_abduction_from_vertical(right_shoulder, right_elbow)
        avg_abduction = (left_abduction + right_abduction) / 2
        
        torso_tilt = calculate_torso_tilt_from_vertical(left_shoulder, right_shoulder, left_hip, right_hip)
        
        elbow_angle_left = calculate_elbow_extension_angle(left_shoulder, left_elbow, left_wrist)
        elbow_angle_right = calculate_elbow_extension_angle(right_shoulder, right_elbow, right_wrist)
        
        # Shoulder elevation
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        if session_data.baseline_hip_y == 0:
            session_data.baseline_hip_y = (left_hip.y + right_hip.y) / 2
        shoulder_elevation = abs(shoulder_mid_y - session_data.baseline_hip_y) * 100
        
        # Wrist positions
        wrist_pos_left = (left_wrist.x, left_wrist.y)
        wrist_pos_right = (right_wrist.x, right_wrist.y)
        
        frame_metrics = {
            'left_shoulder_angle': left_abduction,
            'right_shoulder_angle': right_abduction,
            'avg_shoulder_angle': avg_abduction,
            'torso_tilt': torso_tilt,
            'shoulder_elevation': shoulder_elevation,
            'elbow_angle_left': elbow_angle_left,
            'elbow_angle_right': elbow_angle_right,
            'wrist_pos_left': wrist_pos_left,
            'wrist_pos_right': wrist_pos_right,
            'timestamp': timestamp
        }
        
        session_data.frame_data.append(frame_metrics)
        
    except Exception as e:
        debug_print(f"Error processing frame data: {e}")
    
    return frame_metrics

def get_smoothed_angle(new_angle):
    """Apply moving average smoothing to reduce noise in angle measurements"""
    angle_buffer.append(new_angle)
    if len(angle_buffer) < 2:
        return new_angle
    return float(np.mean(angle_buffer))

def calculate_angle_velocity():
    """Calculate the rate of change of angle to detect movement direction"""
    if len(angle_buffer) < 2:
        return 0.0
    return angle_buffer[-1] - angle_buffer[-2]

def update_rep_state(avg_angle, timestamp, frame_metrics):
    """Update rep state machine with improved accuracy and proper up+down = 1 rep counting"""
    global rep_state, current_rep_data, peak_angle_recorded, peak_confirmation_frames, last_rep_completion_time
    
    try:
        # Apply smoothing to reduce noise
        smoothed_angle = get_smoothed_angle(avg_angle)
        angle_velocity = calculate_angle_velocity()
        angle_velocity_buffer.append(angle_velocity)
        
        # Calculate average velocity for smoother detection
        avg_velocity = float(np.mean(list(angle_velocity_buffer))) if len(angle_velocity_buffer) >= 2 else angle_velocity
        
        # Debug: print current state and angle more frequently when tracking is active
        if tracking_active and frame_metrics and int(timestamp * 5) % 5 == 0:  # Every ~0.2 seconds when tracking
            debug_print(f"[TRACKING] State: {rep_state}, Angle: {smoothed_angle:.1f}°, Peak: {peak_angle_recorded:.1f}°, Velocity: {avg_velocity:.2f}")
        
        if rep_state == "idle":
            # Start rep when angle exceeds threshold (simplified - removed strict velocity check)
            if smoothed_angle > THRESHOLDS['rep_start']:
                rep_state = "up_phase"
                current_rep_data = RepData()
                current_rep_data.start_time = timestamp
                current_rep_data.rep_number = len(session_data.reps) + 1
                peak_angle_recorded = smoothed_angle
                peak_confirmation_frames = 0
                
                if session_data.last_rep_end_time > 0:
                    current_rep_data.time_between_reps = timestamp - session_data.last_rep_end_time
                
                debug_print(f"Rep {current_rep_data.rep_number} UP PHASE started at {smoothed_angle:.1f}°")
        
        elif rep_state == "up_phase":
            if current_rep_data:
                # Track the maximum angle reached
                if smoothed_angle > peak_angle_recorded:
                    peak_angle_recorded = smoothed_angle
                    peak_confirmation_frames = 0  # Reset confirmation counter when new peak found
                
                # Check if we've reached the peak zone and angle is starting to decrease
                if peak_angle_recorded >= THRESHOLDS['rep_peak_min']:
                    # Detect if arm has started descending (angle dropping from peak) - lowered to 3 degrees
                    if smoothed_angle < (peak_angle_recorded - 3.0):  # 3 degree drop from peak (more lenient)
                        peak_confirmation_frames += 1
                        
                        # Confirm peak after sustained downward movement
                        if peak_confirmation_frames >= THRESHOLDS['peak_hold_frames']:
                            current_rep_data.peak_time = timestamp
                            current_rep_data.rom_peak_angle = peak_angle_recorded
                            rep_state = "down_phase"
                            debug_print(f"Rep {current_rep_data.rep_number} PEAK confirmed at {peak_angle_recorded:.1f}° - starting DOWN PHASE")
                
                # Safety: if angle drops significantly without reaching peak, reset (more lenient)
                if smoothed_angle < 8.0 and peak_angle_recorded < THRESHOLDS['rep_peak_min']:
                    debug_print(f"Rep {current_rep_data.rep_number} ABORTED - dropped below start threshold without reaching peak (peak was {peak_angle_recorded:.1f}°)")
                    rep_state = "idle"
                    current_rep_data = None
                    peak_angle_recorded = 0.0
                    peak_confirmation_frames = 0
        
        elif rep_state == "down_phase":
            if current_rep_data:
                # Complete rep when angle drops below end threshold
                if smoothed_angle < THRESHOLDS['rep_end']:
                    # Verify minimum rep duration to filter out noise/false positives
                    rep_duration = timestamp - current_rep_data.start_time
                    
                    if rep_duration >= THRESHOLDS['min_rep_duration']:
                        current_rep_data.end_time = timestamp
                        session_data.last_rep_end_time = timestamp
                        last_rep_completion_time = timestamp
                        
                        debug_print(f"Rep {current_rep_data.rep_number} COMPLETED! (Duration: {rep_duration:.2f}s, Peak: {peak_angle_recorded:.1f}°)")
                        
                        process_completed_rep()
                        rep_state = "idle"
                        current_rep_data = None
                        peak_angle_recorded = 0.0
                        peak_confirmation_frames = 0
                    else:
                        debug_print(f"Rep {current_rep_data.rep_number} rejected - too short ({rep_duration:.2f}s < {THRESHOLDS['min_rep_duration']}s)")
                        rep_state = "idle"
                        current_rep_data = None
                        peak_angle_recorded = 0.0
                        peak_confirmation_frames = 0
                    
    except Exception as e:
        debug_print(f"Error in rep state update: {e}")

def process_completed_rep():
    """Process completed rep"""
    global current_rep_data, current_db_session_id
    
    if not current_rep_data:
        return
    
    try:
        # Extract data from session frame data (last few frames)
        recent_frames = session_data.frame_data[-20:] if len(session_data.frame_data) >= 20 else session_data.frame_data
        
        for frame in recent_frames:
            current_rep_data.left_angles.append(frame.get('left_shoulder_angle', 0))
            current_rep_data.right_angles.append(frame.get('right_shoulder_angle', 0))
            current_rep_data.avg_angles.append(frame.get('avg_shoulder_angle', 0))
            current_rep_data.timestamps.append(frame.get('timestamp', 0))
            current_rep_data.torso_tilts.append(frame.get('torso_tilt', 0))
            current_rep_data.shoulder_elevations.append(frame.get('shoulder_elevation', 0))
            current_rep_data.elbow_angles_left.append(frame.get('elbow_angle_left', 170))
            current_rep_data.elbow_angles_right.append(frame.get('elbow_angle_right', 170))
            current_rep_data.left_wrist_trajectory.append(frame.get('wrist_pos_left', (0, 0)))
            current_rep_data.right_wrist_trajectory.append(frame.get('wrist_pos_right', (0, 0)))
        
        # Calculate timing
        current_rep_data.rep_duration = current_rep_data.end_time - current_rep_data.start_time
        
        # ROM calculations
        if current_rep_data.avg_angles:
            current_rep_data.rom_peak_angle = max(current_rep_data.avg_angles)
            current_rep_data.rom_min_angle = min(current_rep_data.avg_angles)
            current_rep_data.rom_total = current_rep_data.rom_peak_angle - current_rep_data.rom_min_angle
        
        # Bilateral symmetry
        if current_rep_data.left_angles and current_rep_data.right_angles:
            left_max = max(current_rep_data.left_angles)
            right_max = max(current_rep_data.right_angles)
            current_rep_data.symmetry_diff = abs(left_max - right_max)
        
        # Stability metrics
        if current_rep_data.torso_tilts:
            current_rep_data.torso_lean_max = max(current_rep_data.torso_tilts)
        
        # Elbow assessment
        all_elbow_angles = current_rep_data.elbow_angles_left + current_rep_data.elbow_angles_right
        if all_elbow_angles:
            current_rep_data.elbow_angle_avg = float(np.mean(all_elbow_angles))
        
        # Calculate scores
        calculate_rep_scores(current_rep_data)
        
        # Add to session
        session_data.rep_durations.append(current_rep_data.rep_duration)
        session_data.reps.append(current_rep_data)
        
        # Save to database if we have an active session
        if current_db_session_id:
            try:
                save_rep(
                    session_id=current_db_session_id,
                    rep_number=current_rep_data.rep_number,
                    score=current_rep_data.score_overall,
                    grade=current_rep_data.form_grade,
                    rom_angle=current_rep_data.rom_total,
                    symmetry_diff=current_rep_data.symmetry_diff,
                    torso_stability=current_rep_data.score_torso_stability * 100,
                    elbow_angle=current_rep_data.elbow_angle_avg,
                    duration=current_rep_data.rep_duration,
                    warnings=current_rep_data.warnings
                )
                
                # Check for personal records
                check_personal_records(1, current_rep_data.score_overall, current_rep_data.rom_total)
                debug_print(f"Rep {current_rep_data.rep_number} saved to database")
            except Exception as db_error:
                debug_print(f"Database save error: {db_error}")
        
        # Add to chat history
        chat_message = f"✅ Rep {current_rep_data.rep_number} completed! Score: {current_rep_data.score_overall:.1f}/100 | ROM: {current_rep_data.rom_total:.1f}° | Grade: {current_rep_data.form_grade}"
        chat_history.append({
            'role': 'system',
            'message': chat_message,
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S')
        })
        
        debug_print(f"Rep {current_rep_data.rep_number} analyzed - Score: {current_rep_data.score_overall:.1f}")
        
    except Exception as e:
        debug_print(f"Error processing completed rep: {e}")

def calculate_rep_scores(rep_data):
    """Calculate rep scores"""
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
        
        # Smoothness Score (simplified)
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
        weights = [0.25, 0.20, 0.20, 0.15, 0.20]  # ROM, Symmetry, Torso, Smoothness, Elbow
        scores = [
            rep_data.score_rom,
            rep_data.score_symmetry,
            rep_data.score_torso_stability,
            rep_data.score_smoothness,
            rep_data.score_elbow_position
        ]
        
        rep_data.score_overall = sum(s * w for s, w in zip(scores, weights)) * 100
        
        # Assign grade
        score = rep_data.score_overall
        if score >= 90: rep_data.form_grade = 'A+'
        elif score >= 85: rep_data.form_grade = 'A'
        elif score >= 80: rep_data.form_grade = 'A-'
        elif score >= 75: rep_data.form_grade = 'B+'
        elif score >= 70: rep_data.form_grade = 'B'
        elif score >= 65: rep_data.form_grade = 'B-'
        elif score >= 60: rep_data.form_grade = 'C+'
        elif score >= 55: rep_data.form_grade = 'C'
        else: rep_data.form_grade = 'F'
        
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
        debug_print(f"Error calculating scores: {e}")
        rep_data.score_overall = 50.0
        rep_data.form_grade = 'C'

def update_trajectory_visualization(landmarks, frame_width, frame_height):
    """Update trajectory buffers"""
    try:
        if not landmarks:
            return
            
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        if left_wrist.visibility < 0.6 or right_wrist.visibility < 0.6:
            return
        
        left_point = (int(left_wrist.x * frame_width), int(left_wrist.y * frame_height))
        right_point = (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height))
        
        if (0 <= left_point[0] <= frame_width and 0 <= left_point[1] <= frame_height):
            left_wrist_trajectory.append(left_point)
            
        if (0 <= right_point[0] <= frame_width and 0 <= right_point[1] <= frame_height):
            right_wrist_trajectory.append(right_point)
        
        if len(trajectory_colors) >= trajectory_buffer_size:
            trajectory_colors.pop(0)
        
        progress = len(left_wrist_trajectory) / trajectory_buffer_size
        color = tuple(int(c1 + progress * (c2 - c1)) for c1, c2 in zip(COLORS['trajectory_start'], COLORS['trajectory_end']))
        trajectory_colors.append(color)
        
    except Exception as e:
        debug_print(f"Error updating trajectory: {e}")

def draw_trajectory_path(img, trajectory, colors, thickness=2):
    """Draw trajectory path"""
    try:
        if len(trajectory) < 2:
            return
        
        for i in range(1, len(trajectory)):
            if i < len(colors):
                color = colors[i]
            else:
                color = COLORS['trajectory_end']
            
            segment_thickness = max(1, thickness - (len(trajectory) - i) // 15)
            cv2.line(img, trajectory[i-1], trajectory[i], color, segment_thickness)
            
            if i == len(trajectory) - 1:
                cv2.circle(img, trajectory[i], 6, COLORS['trajectory_end'], -1)
                cv2.circle(img, trajectory[i], 8, COLORS['white'], 2)
            
    except Exception as e:
        debug_print(f"Error drawing trajectory: {e}")

def start_camera():
    """Start camera capture"""
    global camera_active, cap, current_frame, current_mesh_frame
    
    try:
        # Try platform-appropriate backend
        import platform
        system = platform.system()
        
        if system == 'Windows':
            # Try DirectShow backend first on Windows
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                debug_print("DirectShow failed, trying default backend...")
                cap = cv2.VideoCapture(0)
        elif system == 'Darwin':  # macOS
            # Use AVFoundation on Mac (default) with optimized settings
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                debug_print("AVFoundation failed, trying default backend...")
                cap = cv2.VideoCapture(0)
        else:
            # Linux or other
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            debug_print("Error: Camera not accessible")
            return False
        
        # Set camera properties - use lower resolution for better performance on Mac
        if system == 'Darwin':
            # Lower resolution for Mac to improve frame rate
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        # Verify camera is actually capturing
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            debug_print("Error: Camera opened but cannot read frames")
            cap.release()
            return False
        
        camera_active = True
        debug_print(f"✅ Camera started successfully on {system}")
        return True
        
    except Exception as e:
        debug_print(f"❌ Camera initialization error: {e}")
        return False

def camera_loop():
    """Main camera processing loop"""
    global current_frame, current_mesh_frame, frame_counter
    
    import platform
    is_mac = platform.system() == 'Darwin'
    
    calibration_frames = 0
    calibration_complete = False
    
    # Use lighter model on Mac for better performance
    pose_complexity = 0 if is_mac else 1  # 0 = lite, 1 = full
    detection_confidence = 0.6 if is_mac else 0.75
    tracking_confidence = 0.6 if is_mac else 0.75
    
    # Skip segmentation on Mac for better performance
    use_segmentation = not is_mac
    
    seg_context = mp_selfie.SelfieSegmentation(model_selection=1) if use_segmentation else None
    
    with mp_pose.Pose(static_image_mode=False,
                      model_complexity=pose_complexity,
                      min_detection_confidence=detection_confidence,
                      min_tracking_confidence=tracking_confidence) as pose:
        
        debug_print(f"✅ MediaPipe initialized (Mac optimized: {is_mac}) - Camera processing started")
        
        if seg_context:
            seg_context.__enter__()
        
        try:
            while camera_active and cap and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.001)
                        continue
                    
                    height, width = frame.shape[:2]
                    frame_counter += 1
                    timestamp = frame_counter * 0.033
                    
                    # Create mesh frame (black background)
                    mesh_frame = np.zeros_like(frame)
                    
                    # Convert to RGB once
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Background blurring (skip on Mac for performance)
                    human_detected = True
                    if use_segmentation and seg_context:
                        try:
                            seg_results = seg_context.process(rgb_frame)
                            
                            if seg_results and seg_results.segmentation_mask is not None:
                                mask = seg_results.segmentation_mask
                                blurred = cv2.GaussianBlur(frame, (15, 15), 0)
                                condition = mask > 0.3
                                condition_3 = np.dstack((condition, condition, condition))
                                composed = np.where(condition_3, frame, blurred).astype(np.uint8)
                                human_detected = np.any(condition)
                            else:
                                composed = frame.copy()
                                human_detected = False
                        except Exception:
                            composed = frame.copy()
                            human_detected = True
                    else:
                        # No segmentation - just use the frame directly (faster)
                        composed = frame.copy()
                    
                    # Pose processing
                    frame_metrics = None
                    
                    if human_detected:
                        try:
                            pose_results = pose.process(rgb_frame)
                            
                            if pose_results and pose_results.pose_landmarks:
                                landmarks = pose_results.pose_landmarks.landmark
                                
                                # Calibration phase
                                if not calibration_complete and calibration_frames < 60:
                                    calibration_frames += 1
                                    cv2.putText(composed, f"INITIALIZING... {calibration_frames}/60", 
                                               (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                               0.8, (0, 255, 0), 2, cv2.LINE_AA)
                                    
                                    if calibration_frames >= 60:
                                        calibration_complete = True
                                        debug_print("✅ System calibration complete!")
                                
                                # Process frame data after calibration
                                if calibration_complete:
                                    frame_metrics = process_frame_data(landmarks, frame_counter, timestamp)
                                    
                                    # Update trajectory visualization
                                    update_trajectory_visualization(landmarks, width, height)
                                    
                                    # Rep state processing
                                    if tracking_active and frame_metrics:
                                        update_rep_state(frame_metrics['avg_shoulder_angle'], timestamp, frame_metrics)
                                
                                # Draw pose landmarks
                                try:
                                    mp_drawing.draw_landmarks(
                                        composed, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3),
                                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=2)
                                    )
                                    
                                    mp_drawing.draw_landmarks(
                                        mesh_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=5),
                                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=4)
                                    )
                                    
                                    # Add key joint highlights
                                    key_landmarks = [
                                        mp_pose.PoseLandmark.LEFT_SHOULDER,
                                        mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                        mp_pose.PoseLandmark.LEFT_ELBOW,
                                        mp_pose.PoseLandmark.RIGHT_ELBOW,
                                        mp_pose.PoseLandmark.LEFT_WRIST,
                                        mp_pose.PoseLandmark.RIGHT_WRIST
                                    ]
                                    
                                    for landmark in key_landmarks:
                                        point = landmarks[landmark.value]
                                        if point.visibility > 0.6:
                                            x, y = int(point.x * width), int(point.y * height)
                                            cv2.circle(composed, (x, y), 8, (255, 255, 0), -1)
                                            cv2.circle(composed, (x, y), 10, (0, 255, 0), 2)
                                            cv2.circle(mesh_frame, (x, y), 10, (255, 255, 0), -1)
                                            cv2.circle(mesh_frame, (x, y), 12, (0, 255, 0), 3)
                                    
                                except Exception as draw_err:
                                    debug_print(f"Drawing landmarks error: {draw_err}")
                                
                                # Draw trajectory paths
                                if calibration_complete and len(trajectory_colors) > 0:
                                    draw_trajectory_path(composed, left_wrist_trajectory, trajectory_colors, thickness=3)
                                    draw_trajectory_path(composed, right_wrist_trajectory, trajectory_colors, thickness=3)
                                    draw_trajectory_path(mesh_frame, left_wrist_trajectory, trajectory_colors, thickness=5)
                                    draw_trajectory_path(mesh_frame, right_wrist_trajectory, trajectory_colors, thickness=5)
                        
                        except Exception as pose_err:
                            debug_print(f"Pose processing error: {pose_err}")
                    
                    # Add status overlay
                    cv2.rectangle(composed, (10, 10), (400, 100), (0, 0, 0), -1)
                    cv2.rectangle(composed, (10, 10), (400, 100), (0, 255, 0), 2)
                    
                    cv2.putText(composed, f"SHOULDER RAISE ANALYZER", 
                               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    rep_text = f"REPS: {len(session_data.reps)}"
                    if session_data.reps:
                        latest_grade = session_data.reps[-1].form_grade
                        rep_text += f" | LAST: {latest_grade}"
                    cv2.putText(composed, rep_text, 
                               (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    state_color = (0, 255, 0) if tracking_active else (100, 100, 100)
                    state_text = f"STATE: {rep_state.upper()}" if tracking_active else "STATE: STANDBY"
                    cv2.putText(composed, state_text, 
                               (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2, cv2.LINE_AA)
                    
                    # Add system info to mesh frame
                    cv2.rectangle(mesh_frame, (10, 10), (400, 60), (0, 0, 0), -1)
                    cv2.rectangle(mesh_frame, (10, 10), (400, 60), (0, 255, 0), 2)
                    cv2.putText(mesh_frame, "POSE ANALYSIS VIEW", 
                               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Update global frames
                    current_frame = composed.copy()
                    current_mesh_frame = mesh_frame.copy()
                    
                except Exception as loop_err:
                    debug_print(f"Error in camera loop: {loop_err}")
                    time.sleep(0.01)
        
        finally:
            if seg_context:
                seg_context.__exit__(None, None, None)

def generate_frames():
    """Generate frames for video streaming"""
    global current_frame
    
    while True:
        try:
            frame = current_frame
            if frame is not None:
                # Lower quality on Mac for faster encoding
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            debug_print(f"Error encoding frame: {e}")
        time.sleep(0.016)  # ~60 FPS max (reduced from 0.033)

def generate_mesh_frames():
    """Generate mesh frames for video streaming"""
    global current_mesh_frame
    
    while True:
        try:
            frame = current_mesh_frame
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            debug_print(f"Error encoding mesh frame: {e}")
        time.sleep(0.016)  # ~60 FPS max
        time.sleep(0.033)

# Flask Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """Chat page"""
    return render_template('chat.html', chat_history=chat_history)

@app.route('/camera')
def camera():
    """Camera analysis page"""
    return render_template('camera.html')

@app.route('/upload')
def upload():
    """Video upload page"""
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/history')
def history():
    """History page"""
    return render_template('history.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mesh_feed')
def mesh_feed():
    return Response(generate_mesh_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    status_data = {
        'tracking_active': tracking_active,
        'rep_state': rep_state,
        'rep_count': len(session_data.reps),
        'session_duration': (datetime.datetime.now() - session_data.start_time).total_seconds()
    }
    
    # Add current rep details if available
    if session_data.reps:
        latest_rep = session_data.reps[-1]
        status_data['latest_score'] = latest_rep.score_overall
        status_data['latest_grade'] = latest_rep.form_grade
        status_data['latest_rom'] = latest_rep.rom_total
        status_data['latest_symmetry'] = latest_rep.symmetry_diff
    
    # Add session summary
    if session_data.reps:
        scores = [rep.score_overall for rep in session_data.reps]
        status_data['session_avg_score'] = np.mean(scores) if scores else 0
        status_data['session_best_score'] = max(scores) if scores else 0
        status_data['total_reps'] = len(session_data.reps)
    
    return jsonify(status_data)

@app.route('/api/start_tracking', methods=['POST'])
def start_tracking():
    """Start tracking"""
    global tracking_active, session_data, current_db_session_id
    global peak_angle_recorded, peak_confirmation_frames, last_rep_completion_time, rep_state, current_rep_data
    
    try:
        if tracking_active:
            return jsonify({'success': False, 'error': 'Already tracking'})
        
        tracking_active = True
        session_data = SessionData()
        
        # Reset all rep tracking state for clean start
        rep_state = "idle"
        current_rep_data = None
        peak_angle_recorded = 0.0
        peak_confirmation_frames = 0
        last_rep_completion_time = 0.0
        
        # Clear buffers
        angle_buffer.clear()
        angle_velocity_buffer.clear()
        
        # Create database session
        current_db_session_id = create_session(user_id=1)
        debug_print(f"Created database session: {current_db_session_id}")
        
        # Add welcome message to chat
        chat_history.append({
            'role': 'system',
            'message': '🏋️ Shoulder raise tracking started! Perform lateral raises in front of the camera. One rep = raise arms up + lower them back down.',
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S')
        })
        
        debug_print("Started tracking - Rep counting: UP + DOWN = 1 REP")
        return jsonify({'success': True, 'session_id': current_db_session_id})
        
    except Exception as e:
        debug_print(f"Error starting tracking: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_tracking', methods=['POST'])
def stop_tracking():
    """Stop tracking"""
    global tracking_active, current_db_session_id
    
    try:
        if not tracking_active:
            return jsonify({'success': False, 'error': 'Not currently tracking'})
        
        tracking_active = False
        
        # Complete database session
        if current_db_session_id and session_data.reps:
            scores = [rep.score_overall for rep in session_data.reps]
            avg_score = float(np.mean(scores))
            best_grade = session_data.reps[np.argmax(scores)].form_grade
            complete_session(current_db_session_id, len(session_data.reps), avg_score, best_grade)
            debug_print(f"Completed database session: {current_db_session_id}")
        
        # Add session summary to chat
        if session_data.reps:
            avg_score = np.mean([rep.score_overall for rep in session_data.reps])
            best_score = max([rep.score_overall for rep in session_data.reps])
            best_grade = session_data.reps[-1].form_grade if session_data.reps else 'N/A'
            
            chat_history.append({
                'role': 'system',
                'message': f'✅ Session completed! Total reps: {len(session_data.reps)} | Avg Score: {avg_score:.1f} | Best Score: {best_score:.1f} | Best Grade: {best_grade}',
                'timestamp': datetime.datetime.now().strftime('%H:%M:%S')
            })
        
        current_db_session_id = None
        debug_print("Stopped tracking")
        return jsonify({'success': True})
        
    except Exception as e:
        debug_print(f"Error stopping tracking: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset session"""
    global session_data, rep_state, current_rep_data, tracking_active
    global peak_angle_recorded, peak_confirmation_frames, last_rep_completion_time
    
    try:
        session_data = SessionData()
        rep_state = "idle"
        current_rep_data = None
        tracking_active = False
        
        # Reset rep tracking state
        peak_angle_recorded = 0.0
        peak_confirmation_frames = 0
        last_rep_completion_time = 0.0
        
        # Clear buffers
        angle_buffer.clear()
        angle_velocity_buffer.clear()
        
        # Clear trajectory data
        left_wrist_trajectory.clear()
        right_wrist_trajectory.clear()
        trajectory_colors.clear()
        
        debug_print("Session reset")
        return jsonify({'success': True})
        
    except Exception as e:
        debug_print(f"Error resetting session: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chat/send', methods=['POST'])
def send_chat_message():
    """Send chat message"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request data'})
        
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'success': False, 'error': 'Message is required'})
        
        # Input validation - limit message length to prevent abuse
        max_message_length = 1000
        if len(message) > max_message_length:
            return jsonify({'success': False, 'error': f'Message too long (max {max_message_length} characters)'})
        
        # Add user message to chat
        chat_history.append({
            'role': 'user',
            'message': message,
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S')
        })
        
        # Simple chatbot responses for shoulder raise questions
        response = get_chatbot_response(message)
        
        # Add bot response to chat
        chat_history.append({
            'role': 'bot',
            'message': response,
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S')
        })
        
        return jsonify({'success': True, 'response': response})
        
    except Exception as e:
        debug_print(f"Error sending chat message: {e}")
        return jsonify({'success': False, 'error': str(e)})

def get_chatbot_response(message):
    """Simple chatbot for shoulder raise questions"""
    message_lower = message.lower()
    
    # Common shoulder raise questions
    if any(word in message_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm your shoulder raise assistant. How can I help you with your lateral raise form?"
    
    elif any(word in message_lower for word in ['form', 'technique', 'proper']):
        return "Proper lateral raise form: Stand tall, keep elbows slightly bent, raise arms to shoulder height, avoid shrugging shoulders, control the movement both up and down."
    
    elif any(word in message_lower for word in ['common mistake', 'error', 'wrong']):
        return "Common mistakes: 1) Using too much weight 2) Shrugging shoulders 3) Swinging the body 4) Raising arms too high 5) Bending elbows excessively."
    
    elif any(word in message_lower for word in ['elbow', 'bend']):
        return "Keep a slight bend in elbows (160-180°). Don't lock them out completely or bend too much. This protects your joints."
    
    elif any(word in message_lower for word in ['shoulder height', 'how high']):
        return "Raise arms to shoulder height (parallel to ground). Going higher can impinge shoulder joint."
    
    elif any(word in message_lower for word in ['score', 'grading', 'evaluation']):
        return "Your form is evaluated on: ROM (70-90° ideal), symmetry (<5° difference), torso stability, movement smoothness, and elbow position."
    
    elif any(word in message_lower for word in ['pain', 'hurt', 'injury']):
        return "If you feel pain, stop immediately. Reduce weight, check form, and consider consulting a professional. Never work through joint pain."
    
    elif any(word in message_lower for word in ['benefit', 'why', 'purpose']):
        return "Lateral raises target medial deltoids, improving shoulder width and strength. Helps with shoulder stability and posture."
    
    elif any(word in message_lower for word in ['weight', 'heavy', 'light']):
        return "Start light to master form. Weight should allow controlled movement without swinging. Quality over quantity!"
    
    elif any(word in message_lower for word in ['rep', 'count', 'how many']):
        return "Aim for 8-15 reps per set with good form. 3-4 sets total. Rest 60-90 seconds between sets."
    
    elif any(word in message_lower for word in ['thank', 'thanks']):
        return "You're welcome! Keep up the good work. Remember, consistency with proper form is key!"
    
    else:
        return "I'm here to help with shoulder raise questions! Ask me about form, technique, common mistakes, or your performance scores."

@app.route('/api/chat/history')
def get_chat_history():
    """Get chat history"""
    return jsonify({'success': True, 'history': chat_history})

@app.route('/api/session_summary')
def get_session_summary():
    """Get session summary"""
    try:
        if not session_data.reps:
            return jsonify({'success': False, 'error': 'No reps recorded'})
        
        scores = [rep.score_overall for rep in session_data.reps]
        rom_values = [rep.rom_total for rep in session_data.reps]
        symmetry_values = [rep.symmetry_diff for rep in session_data.reps]
        durations = [rep.rep_duration for rep in session_data.reps]
        
        summary = {
            'total_reps': len(session_data.reps),
            'avg_score': float(np.mean(scores)),
            'best_score': float(max(scores)),
            'worst_score': float(min(scores)),
            'avg_rom': float(np.mean(rom_values)),
            'avg_symmetry': float(np.mean(symmetry_values)),
            'avg_duration': float(np.mean(durations)),
            'best_grade': session_data.reps[np.argmax(scores)].form_grade,
            'session_duration': (datetime.datetime.now() - session_data.start_time).total_seconds()
        }
        
        return jsonify({'success': True, 'summary': summary})
        
    except Exception as e:
        debug_print(f"Error getting session summary: {e}")
        return jsonify({'success': False, 'error': str(e)})

# New Dashboard and History API endpoints
@app.route('/api/dashboard/stats')
def get_dashboard_stats():
    """Get dashboard statistics from database"""
    try:
        user_id = 1  # Default user for now
        
        # Get overall stats
        overall_stats = get_overall_stats(user_id)
        
        # Get weekly progress (daily stats)
        weekly_progress = get_weekly_progress(user_id)
        
        # Get recent sessions
        recent_sessions = get_recent_sessions(user_id, limit=10)
        
        # Get personal records and convert to dict format for frontend
        personal_records_list = get_personal_records(user_id)
        personal_records = {}
        for record in personal_records_list:
            record_type = record.get('type', '')
            personal_records[record_type] = {
                'value': record.get('value', 0),
                'achieved_at': record.get('achieved_at')
            }
        
        # Format the response to match frontend expectations
        response = {
            'success': True,
            'stats': {
                'total_reps': overall_stats.get('total_reps', 0),
                'total_sessions': overall_stats.get('total_sessions', 0),
                'avg_score': overall_stats.get('avg_score', 0),
                'best_score': overall_stats.get('best_score', 0),
                'total_workout_time': overall_stats.get('total_workout_time', 0),
                # Frontend expects these specific field names
                'lifetime_avg_score': overall_stats.get('avg_score', 0),
                'all_time_best_score': overall_stats.get('best_score', 0),
                'grade_distribution': overall_stats.get('grade_distribution', {})
            },
            'weekly_progress': weekly_progress,
            'daily_stats': weekly_progress,  # Frontend expects daily_stats for charts
            'recent_sessions': recent_sessions,
            'personal_records': personal_records_list,
            'records': personal_records,  # Frontend expects records as dict
            'grade_distribution': overall_stats.get('grade_distribution', {})
        }
        
        return jsonify(response)
        
    except Exception as e:
        debug_print(f"Error getting dashboard stats: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/history')
def get_history():
    """Get workout history from database"""
    try:
        user_id = 1  # Default user for now
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        sessions = get_session_history(user_id, limit=limit, offset=offset)
        
        # Calculate week and month stats for frontend
        week_stats = get_week_stats(user_id)
        month_stats = get_month_stats(user_id)
        score_trend = get_score_trend(user_id)
        
        return jsonify({
            'success': True,
            'sessions': sessions,
            'count': len(sessions),
            'week_stats': week_stats,
            'month_stats': month_stats,
            'score_trend': score_trend
        })
        
    except Exception as e:
        debug_print(f"Error getting history: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/session/<int:session_id>/reps')
def get_session_reps_api(session_id):
    """Get reps for a specific session"""
    try:
        reps = get_session_reps(session_id)
        
        return jsonify({
            'success': True,
            'reps': reps,
            'count': len(reps)
        })
        
    except Exception as e:
        debug_print(f"Error getting session reps: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ==================== PROFILE ROUTES ====================

@app.route('/profile')
def profile():
    """Profile page"""
    return render_template('profile.html')

@app.route('/leaderboard')
def leaderboard():
    """Leaderboard page"""
    return render_template('leaderboard.html')

@app.route('/api/profile')
def get_profile():
    """Get current user profile"""
    try:
        user_id = session.get('user_id', 1)
        profile = get_user_profile(user_id)
        
        if profile:
            # Get additional stats
            overall_stats = get_overall_stats(user_id)
            achievements = get_user_achievements(user_id)
            rank_info = get_user_rank(user_id)
            
            return jsonify({
                'success': True,
                'profile': profile,
                'stats': overall_stats,
                'achievements': achievements,
                'rank': rank_info
            })
        
        return jsonify({'success': False, 'error': 'Profile not found'})
        
    except Exception as e:
        debug_print(f"Error getting profile: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/profile/update', methods=['POST'])
def update_profile():
    """Update user profile"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        user_id = session.get('user_id', 1)
        
        # Update profile with provided fields
        success = update_user_profile(
            user_id,
            display_name=data.get('display_name'),
            email=data.get('email'),
            avatar_color=data.get('avatar_color'),
            bio=data.get('bio'),
            fitness_goal=data.get('fitness_goal'),
            experience_level=data.get('experience_level')
        )
        
        if success:
            return jsonify({'success': True, 'message': 'Profile updated'})
        return jsonify({'success': False, 'error': 'Failed to update profile'})
        
    except Exception as e:
        debug_print(f"Error updating profile: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/leaderboard')
def get_leaderboard_api():
    """Get leaderboard data"""
    try:
        score_type = request.args.get('type', 'best_score')
        limit = request.args.get('limit', 10, type=int)
        
        leaderboard_data = get_leaderboard(score_type, limit)
        
        # Get current user's rank
        user_id = session.get('user_id', 1)
        user_rank = get_user_rank(user_id, score_type)
        
        return jsonify({
            'success': True,
            'leaderboard': leaderboard_data,
            'user_rank': user_rank,
            'score_type': score_type
        })
        
    except Exception as e:
        debug_print(f"Error getting leaderboard: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/achievements')
def get_achievements_api():
    """Get user achievements"""
    try:
        user_id = session.get('user_id', 1)
        achievements = get_user_achievements(user_id)
        
        return jsonify({
            'success': True,
            'achievements': achievements
        })
        
    except Exception as e:
        debug_print(f"Error getting achievements: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Initialize system
    debug_print("🚀 Starting PosePro Analysis System...")
    
    # Start camera
    if not start_camera():
        debug_print("❌ Camera initialization failed")
        exit(1)
    
    # Start camera processing in background thread
    camera_thread = threading.Thread(target=camera_loop)
    camera_thread.daemon = True
    camera_thread.start()
    
    debug_print("✅ System initialized successfully!")
    debug_print("🌐 Web interface: http://localhost:5005")
    
    try:
        # Disable reloader to prevent killing the camera thread
        app.run(host='0.0.0.0', port=5005, debug=False, threaded=True)
    except KeyboardInterrupt:
        debug_print("🛑 System shutdown by user")
    finally:
        camera_active = False
        if cap:
            cap.release()
        debug_print("✅ Cleanup completed")
