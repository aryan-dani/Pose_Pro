"""
PosePro - AI-Powered Real-Time Shoulder Raise Form Analysis
Main Flask application with route handlers and pose processing loop
"""

# Suppress warnings - MUST be at the very top before any imports
import os
import sys
import warnings
import logging

# Set environment variables BEFORE importing any ML libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

# Filter Python warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*inference_feedback_manager.*')
warnings.filterwarnings('ignore', message='.*SymbolDatabase.GetPrototype.*')

# Temporarily redirect stderr during MediaPipe import
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
import time
import threading
import platform
from flask import Flask, render_template, jsonify, request, Response, session

# Configure absl logging after import
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)
except ImportError:
    pass

# Import local modules from backend package
from backend.config import (
    THRESHOLDS, COLORS, MEDIAPIPE_SETTINGS, 
    SERVER_HOST, SERVER_PORT, SECRET_KEY
)
from backend.models import RepData, SessionData
from backend.pose_analyzer import (
    calculate_shoulder_abduction_from_vertical,
    calculate_torso_tilt_from_vertical,
    calculate_elbow_extension_angle,
    calculate_rep_scores,
    AngleSmoother
)
from backend.chatbot import get_chatbot_response, chat_history
from backend.camera_handler import CameraHandler, draw_trajectory_path, debug_print
from backend.database import (
    init_database, create_session, save_rep, complete_session,
    get_session_history, get_session_reps, check_personal_records,
    get_overall_stats, get_weekly_progress, get_recent_sessions, get_personal_records,
    get_week_stats, get_month_stats, get_score_trend,
    get_user_profile, update_user_profile, create_user,
    get_leaderboard, get_user_rank, get_user_achievements
)

# MediaPipe helpers
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

# Flask app
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Initialize database
init_database()

# Global state
camera_handler = CameraHandler()
tracking_active = False
current_db_session_id = None
session_data = SessionData()
current_rep_data = None
rep_state = "idle"

# Rep tracking state
peak_angle_recorded = 0.0
peak_confirmation_frames = 0
last_rep_completion_time = 0.0
angle_smoother = AngleSmoother()
angle_velocity_buffer = deque(maxlen=5)


def process_frame_data(landmarks, frame_counter, timestamp):
    """Process frame-level data with improved validation."""
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
        
        # Check visibility
        visibility_threshold = MEDIAPIPE_SETTINGS['visibility_threshold']
        if any(lm.visibility < visibility_threshold for lm in [left_shoulder, right_shoulder]):
            return frame_metrics
        
        # Calculate metrics
        left_abduction = calculate_shoulder_abduction_from_vertical(left_shoulder, left_elbow)
        right_abduction = calculate_shoulder_abduction_from_vertical(right_shoulder, right_elbow)
        avg_abduction = (left_abduction + right_abduction) / 2
        
        torso_tilt = calculate_torso_tilt_from_vertical(
            left_shoulder, right_shoulder, left_hip, right_hip
        )
        
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


def update_rep_state(avg_angle, timestamp, frame_metrics):
    """Update rep state machine with improved accuracy."""
    global rep_state, current_rep_data, peak_angle_recorded, peak_confirmation_frames, last_rep_completion_time
    
    try:
        smoothed_angle = angle_smoother.get_smoothed(avg_angle)
        angle_velocity = angle_smoother.get_velocity()
        angle_velocity_buffer.append(angle_velocity)
        
        avg_velocity = float(np.mean(list(angle_velocity_buffer))) if len(angle_velocity_buffer) >= 2 else angle_velocity
        
        if tracking_active and frame_metrics and int(timestamp * 5) % 5 == 0:
            debug_print(f"[TRACKING] State: {rep_state}, Angle: {smoothed_angle:.1f}°, Peak: {peak_angle_recorded:.1f}°, Velocity: {avg_velocity:.2f}")
        
        if rep_state == "idle":
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
                if smoothed_angle > peak_angle_recorded:
                    peak_angle_recorded = smoothed_angle
                    peak_confirmation_frames = 0
                
                if peak_angle_recorded >= THRESHOLDS['rep_peak_min']:
                    if smoothed_angle < (peak_angle_recorded - 3.0):
                        peak_confirmation_frames += 1
                        
                        if peak_confirmation_frames >= THRESHOLDS['peak_hold_frames']:
                            current_rep_data.peak_time = timestamp
                            current_rep_data.rom_peak_angle = peak_angle_recorded
                            rep_state = "down_phase"
                            debug_print(f"Rep {current_rep_data.rep_number} PEAK confirmed at {peak_angle_recorded:.1f}° - starting DOWN PHASE")
                
                if smoothed_angle < 8.0 and peak_angle_recorded < THRESHOLDS['rep_peak_min']:
                    debug_print(f"Rep {current_rep_data.rep_number} ABORTED - dropped below start threshold")
                    rep_state = "idle"
                    current_rep_data = None
                    peak_angle_recorded = 0.0
                    peak_confirmation_frames = 0
        
        elif rep_state == "down_phase":
            if current_rep_data:
                if smoothed_angle < THRESHOLDS['rep_end']:
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
                        debug_print(f"Rep {current_rep_data.rep_number} rejected - too short")
                        rep_state = "idle"
                        current_rep_data = None
                        peak_angle_recorded = 0.0
                        peak_confirmation_frames = 0
                    
    except Exception as e:
        debug_print(f"Error in rep state update: {e}")


def process_completed_rep():
    """Process completed rep and save to database."""
    global current_rep_data
    
    if not current_rep_data:
        return
    
    try:
        # Extract data from session frame data
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
        
        # Save to database
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
                check_personal_records(1, current_rep_data.score_overall, current_rep_data.rom_total)
                debug_print(f"Rep {current_rep_data.rep_number} saved to database")
            except Exception as db_error:
                debug_print(f"Database save error: {db_error}")
        
        # Add to chat history
        chat_message = f"✅ Rep {current_rep_data.rep_number} completed! Score: {current_rep_data.score_overall:.1f}/100 | ROM: {current_rep_data.rom_total:.1f}° | Grade: {current_rep_data.form_grade}"
        chat_history.add_message('system', chat_message)
        
        debug_print(f"Rep {current_rep_data.rep_number} analyzed - Score: {current_rep_data.score_overall:.1f}")
        
    except Exception as e:
        debug_print(f"Error processing completed rep: {e}")


def camera_loop():
    """Main camera processing loop."""
    is_mac = platform.system() == 'Darwin'
    
    calibration_frames = 0
    calibration_complete = False
    
    pose_complexity = MEDIAPIPE_SETTINGS['pose_complexity_mac'] if is_mac else MEDIAPIPE_SETTINGS['pose_complexity_default']
    detection_confidence = MEDIAPIPE_SETTINGS['detection_confidence_mac'] if is_mac else MEDIAPIPE_SETTINGS['detection_confidence_default']
    tracking_confidence = MEDIAPIPE_SETTINGS['tracking_confidence_mac'] if is_mac else MEDIAPIPE_SETTINGS['tracking_confidence_default']
    
    use_segmentation = not is_mac
    seg_context = mp_selfie.SelfieSegmentation(model_selection=1) if use_segmentation else None
    
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=pose_complexity,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence
    ) as pose:
        
        debug_print(f"✅ MediaPipe initialized (Mac optimized: {is_mac})")
        
        if seg_context:
            seg_context.__enter__()
        
        try:
            while camera_handler.camera_active and camera_handler.cap and camera_handler.cap.isOpened():
                try:
                    ret, frame = camera_handler.read_frame()
                    if not ret:
                        time.sleep(0.001)
                        continue
                    
                    height, width = frame.shape[:2]
                    camera_handler.frame_counter += 1
                    timestamp = camera_handler.frame_counter * 0.033
                    
                    mesh_frame = np.zeros_like(frame)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Background blurring
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
                                    frame_metrics = process_frame_data(
                                        landmarks, camera_handler.frame_counter, timestamp
                                    )
                                    camera_handler.update_trajectory(landmarks, width, height, mp_pose)
                                    
                                    if tracking_active and frame_metrics:
                                        update_rep_state(
                                            frame_metrics['avg_shoulder_angle'], 
                                            timestamp, 
                                            frame_metrics
                                        )
                                
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
                                    
                                    # Key joint highlights
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
                                if calibration_complete and len(camera_handler.trajectory_colors) > 0:
                                    draw_trajectory_path(composed, camera_handler.left_wrist_trajectory, camera_handler.trajectory_colors, thickness=3)
                                    draw_trajectory_path(composed, camera_handler.right_wrist_trajectory, camera_handler.trajectory_colors, thickness=3)
                                    draw_trajectory_path(mesh_frame, camera_handler.left_wrist_trajectory, camera_handler.trajectory_colors, thickness=5)
                                    draw_trajectory_path(mesh_frame, camera_handler.right_wrist_trajectory, camera_handler.trajectory_colors, thickness=5)
                        
                        except Exception as pose_err:
                            debug_print(f"Pose processing error: {pose_err}")
                    
                    # Status overlay
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
                    
                    # Mesh frame overlay
                    cv2.rectangle(mesh_frame, (10, 10), (400, 60), (0, 0, 0), -1)
                    cv2.rectangle(mesh_frame, (10, 10), (400, 60), (0, 255, 0), 2)
                    cv2.putText(mesh_frame, "POSE ANALYSIS VIEW", 
                               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Update global frames
                    camera_handler.current_frame = composed.copy()
                    camera_handler.current_mesh_frame = mesh_frame.copy()
                    
                except Exception as loop_err:
                    debug_print(f"Error in camera loop: {loop_err}")
                    time.sleep(0.01)
        
        finally:
            if seg_context:
                seg_context.__exit__(None, None, None)


# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/chat')
def chat_page():
    """Chat page."""
    return render_template('chat.html', chat_history=chat_history.get_history())


@app.route('/camera')
def camera():
    """Camera analysis page."""
    return render_template('camera.html')


@app.route('/upload')
def upload():
    """Video upload page."""
    return render_template('upload.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard page."""
    return render_template('dashboard.html')


@app.route('/history')
def history():
    """History page."""
    return render_template('history.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(camera_handler.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/mesh_feed')
def mesh_feed():
    """Mesh video streaming route."""
    return Response(camera_handler.generate_mesh_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def get_status():
    """Get current system status."""
    status_data = {
        'tracking_active': tracking_active,
        'rep_state': rep_state,
        'rep_count': len(session_data.reps),
        'session_duration': (datetime.datetime.now() - session_data.start_time).total_seconds()
    }
    
    if session_data.reps:
        latest_rep = session_data.reps[-1]
        status_data['latest_score'] = latest_rep.score_overall
        status_data['latest_grade'] = latest_rep.form_grade
        status_data['latest_rom'] = latest_rep.rom_total
        status_data['latest_symmetry'] = latest_rep.symmetry_diff
        
        scores = [rep.score_overall for rep in session_data.reps]
        status_data['session_avg_score'] = np.mean(scores) if scores else 0
        status_data['session_best_score'] = max(scores) if scores else 0
        status_data['total_reps'] = len(session_data.reps)
    
    return jsonify(status_data)


@app.route('/api/start_tracking', methods=['POST'])
def start_tracking():
    """Start tracking (also starts camera if not running)."""
    global tracking_active, session_data, current_db_session_id
    global peak_angle_recorded, peak_confirmation_frames, last_rep_completion_time, rep_state, current_rep_data
    
    try:
        if tracking_active:
            return jsonify({'success': False, 'error': 'Already tracking'})
        
        # Start camera if not already running
        if not camera_handler.camera_active:
            if not camera_handler.start():
                return jsonify({'success': False, 'error': 'Camera initialization failed'})
            
            # Start camera processing thread
            camera_thread = threading.Thread(target=camera_loop)
            camera_thread.daemon = True
            camera_thread.start()
            debug_print("Camera started on-demand")
        
        tracking_active = True
        session_data = SessionData()
        
        # Reset tracking state
        rep_state = "idle"
        current_rep_data = None
        peak_angle_recorded = 0.0
        peak_confirmation_frames = 0
        last_rep_completion_time = 0.0
        
        # Clear buffers
        angle_smoother.clear()
        angle_velocity_buffer.clear()
        
        # Create database session
        current_db_session_id = create_session(user_id=1)
        debug_print(f"Created database session: {current_db_session_id}")
        
        chat_history.add_message('system', '🏋️ Shoulder raise tracking started! Perform lateral raises in front of the camera.')
        
        debug_print("Started tracking - Rep counting: UP + DOWN = 1 REP")
        return jsonify({'success': True, 'session_id': current_db_session_id})
        
    except Exception as e:
        debug_print(f"Error starting tracking: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/stop_tracking', methods=['POST'])
def stop_tracking():
    """Stop tracking and camera."""
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
        
        # Session summary
        if session_data.reps:
            avg_score = np.mean([rep.score_overall for rep in session_data.reps])
            best_score = max([rep.score_overall for rep in session_data.reps])
            best_grade = session_data.reps[-1].form_grade if session_data.reps else 'N/A'
            
            chat_history.add_message(
                'system', 
                f'✅ Session completed! Total reps: {len(session_data.reps)} | Avg Score: {avg_score:.1f} | Best Score: {best_score:.1f}'
            )
        
        # Stop the camera
        camera_handler.stop()
        debug_print("Camera stopped")
        
        current_db_session_id = None
        debug_print("Stopped tracking")
        return jsonify({'success': True})
        
    except Exception as e:
        debug_print(f"Error stopping tracking: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset session."""
    global session_data, rep_state, current_rep_data, tracking_active
    global peak_angle_recorded, peak_confirmation_frames, last_rep_completion_time
    
    try:
        session_data = SessionData()
        rep_state = "idle"
        current_rep_data = None
        tracking_active = False
        
        peak_angle_recorded = 0.0
        peak_confirmation_frames = 0
        last_rep_completion_time = 0.0
        
        angle_smoother.clear()
        angle_velocity_buffer.clear()
        camera_handler.clear_trajectory()
        
        debug_print("Session reset")
        return jsonify({'success': True})
        
    except Exception as e:
        debug_print(f"Error resetting session: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/chat/send', methods=['POST'])
def send_chat_message():
    """Send chat message."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request data'})
        
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'success': False, 'error': 'Message is required'})
        
        max_message_length = 1000
        if len(message) > max_message_length:
            return jsonify({'success': False, 'error': f'Message too long (max {max_message_length} characters)'})
        
        chat_history.add_message('user', message)
        response = get_chatbot_response(message)
        chat_history.add_message('bot', response)
        
        return jsonify({'success': True, 'response': response})
        
    except Exception as e:
        debug_print(f"Error sending chat message: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/chat/history')
def get_chat_history():
    """Get chat history."""
    return jsonify({'success': True, 'history': chat_history.get_history()})


@app.route('/api/session_summary')
def get_session_summary():
    """Get session summary."""
    try:
        summary = session_data.get_summary()
        if not summary:
            return jsonify({'success': False, 'error': 'No reps recorded'})
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        debug_print(f"Error getting session summary: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/dashboard/stats')
def get_dashboard_stats():
    """Get dashboard statistics from database."""
    try:
        user_id = 1
        
        overall_stats = get_overall_stats(user_id)
        weekly_progress = get_weekly_progress(user_id)
        recent_sessions = get_recent_sessions(user_id, limit=10)
        personal_records_list = get_personal_records(user_id)
        
        personal_records = {}
        for record in personal_records_list:
            record_type = record.get('type', '')
            personal_records[record_type] = {
                'value': record.get('value', 0),
                'achieved_at': record.get('achieved_at')
            }
        
        response = {
            'success': True,
            'stats': {
                'total_reps': overall_stats.get('total_reps', 0),
                'total_sessions': overall_stats.get('total_sessions', 0),
                'avg_score': overall_stats.get('avg_score', 0),
                'best_score': overall_stats.get('best_score', 0),
                'total_workout_time': overall_stats.get('total_workout_time', 0),
                'lifetime_avg_score': overall_stats.get('avg_score', 0),
                'all_time_best_score': overall_stats.get('best_score', 0),
                'grade_distribution': overall_stats.get('grade_distribution', {})
            },
            'weekly_progress': weekly_progress,
            'daily_stats': weekly_progress,
            'recent_sessions': recent_sessions,
            'personal_records': personal_records_list,
            'records': personal_records,
            'grade_distribution': overall_stats.get('grade_distribution', {})
        }
        
        return jsonify(response)
        
    except Exception as e:
        debug_print(f"Error getting dashboard stats: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/history')
def get_history():
    """Get workout history from database."""
    try:
        user_id = 1
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        sessions = get_session_history(user_id, limit=limit, offset=offset)
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
    """Get reps for a specific session."""
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
    """Profile page."""
    return render_template('profile.html')


@app.route('/leaderboard')
def leaderboard():
    """Leaderboard page."""
    return render_template('leaderboard.html')


@app.route('/api/profile')
def get_profile():
    """Get current user profile."""
    try:
        user_id = session.get('user_id', 1)
        profile = get_user_profile(user_id)
        
        if profile:
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
    """Update user profile."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        user_id = session.get('user_id', 1)
        
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
    """Get leaderboard data."""
    try:
        score_type = request.args.get('type', 'best_score')
        limit = request.args.get('limit', 10, type=int)
        
        leaderboard_data = get_leaderboard(score_type, limit)
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
    """Get user achievements."""
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


# ==================== MAIN ====================

if __name__ == '__main__':
    debug_print("🚀 Starting PosePro Analysis System...")
    
    # Camera will be started on-demand when user clicks Start on camera page
    debug_print("✅ System initialized (camera will start on-demand)")
    debug_print(f"🌐 Web interface: http://localhost:{SERVER_PORT}")
    
    try:
        app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        debug_print("🛑 System shutdown by user")
    finally:
        camera_handler.stop()
        debug_print("✅ Cleanup completed")

