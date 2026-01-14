"""
Camera Handler module for PosePro
Manages camera initialization, capture, and video streaming
"""

import os
import sys
import warnings
import platform
import time
import threading
import cv2
import numpy as np
from collections import deque

from .config import (
    COLORS, THRESHOLDS, CAMERA_SETTINGS, MEDIAPIPE_SETTINGS,
    TRAJECTORY_BUFFER_SIZE
)


def debug_print(message):
    """Print debug message with timestamp."""
    import datetime
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")


class CameraHandler:
    """Handles camera capture and frame processing."""
    
    def __init__(self):
        self.cap = None
        self.camera_active = False
        self.current_frame = None
        self.current_mesh_frame = None
        self.frame_counter = 0
        
        # Initialize with placeholder frames showing "click start"
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Click START to begin", (130, 230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(placeholder, "Camera will activate", (145, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        self.current_frame = placeholder.copy()
        self.current_mesh_frame = placeholder.copy()
        
        # Trajectory tracking
        self.left_wrist_trajectory = deque(maxlen=TRAJECTORY_BUFFER_SIZE)
        self.right_wrist_trajectory = deque(maxlen=TRAJECTORY_BUFFER_SIZE)
        self.trajectory_colors = []
    
    def start(self):
        """Start camera capture."""
        try:
            system = platform.system()
            
            if system == 'Windows':
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    debug_print("DirectShow failed, trying default backend...")
                    self.cap = cv2.VideoCapture(0)
            elif system == 'Darwin':  # macOS
                self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
                if not self.cap.isOpened():
                    debug_print("AVFoundation failed, trying default backend...")
                    self.cap = cv2.VideoCapture(0)
            else:
                self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                debug_print("Error: Camera not accessible")
                return False
            
            # Set camera properties
            if system == 'Darwin':
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['mac_width'])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['mac_height'])
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['default_width'])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['default_height'])
            
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps'])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_SETTINGS['buffer_size'])
            
            # Verify camera
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                debug_print("Error: Camera opened but cannot read frames")
                self.cap.release()
                return False
            
            self.camera_active = True
            debug_print(f"✅ Camera started successfully on {system}")
            return True
            
        except Exception as e:
            debug_print(f"❌ Camera initialization error: {e}")
            return False
    
    def stop(self):
        """Stop camera capture and reset to placeholder."""
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Reset to placeholder frames
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Click START to begin", (130, 230), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(placeholder, "Camera will activate", (145, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        self.current_frame = placeholder.copy()
        self.current_mesh_frame = placeholder.copy()
    
    def read_frame(self):
        """Read a frame from the camera."""
        if not self.cap or not self.cap.isOpened():
            return False, None
        return self.cap.read()
    
    def update_trajectory(self, landmarks, frame_width, frame_height, mp_pose):
        """Update trajectory buffers for wrist visualization."""
        try:
            if not landmarks:
                return
                
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            
            if left_wrist.visibility < 0.6 or right_wrist.visibility < 0.6:
                return
            
            left_point = (int(left_wrist.x * frame_width), int(left_wrist.y * frame_height))
            right_point = (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height))
            
            if 0 <= left_point[0] <= frame_width and 0 <= left_point[1] <= frame_height:
                self.left_wrist_trajectory.append(left_point)
                
            if 0 <= right_point[0] <= frame_width and 0 <= right_point[1] <= frame_height:
                self.right_wrist_trajectory.append(right_point)
            
            if len(self.trajectory_colors) >= TRAJECTORY_BUFFER_SIZE:
                self.trajectory_colors.pop(0)
            
            progress = len(self.left_wrist_trajectory) / TRAJECTORY_BUFFER_SIZE
            color = tuple(
                int(c1 + progress * (c2 - c1)) 
                for c1, c2 in zip(COLORS['trajectory_start'], COLORS['trajectory_end'])
            )
            self.trajectory_colors.append(color)
            
        except Exception as e:
            debug_print(f"Error updating trajectory: {e}")
    
    def clear_trajectory(self):
        """Clear trajectory visualization data."""
        self.left_wrist_trajectory.clear()
        self.right_wrist_trajectory.clear()
        self.trajectory_colors.clear()
    
    def generate_frames(self):
        """Generator for video streaming frames."""
        while True:
            try:
                frame = self.current_frame
                if frame is not None:
                    ret, buffer = cv2.imencode(
                        '.jpg', frame, 
                        [cv2.IMWRITE_JPEG_QUALITY, CAMERA_SETTINGS['jpeg_quality']]
                    )
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                debug_print(f"Error encoding frame: {e}")
            time.sleep(0.016)  # ~60 FPS max
    
    def generate_mesh_frames(self):
        """Generator for mesh video streaming frames."""
        while True:
            try:
                frame = self.current_mesh_frame
                if frame is not None:
                    ret, buffer = cv2.imencode(
                        '.jpg', frame, 
                        [cv2.IMWRITE_JPEG_QUALITY, CAMERA_SETTINGS['jpeg_quality']]
                    )
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                debug_print(f"Error encoding mesh frame: {e}")
            time.sleep(0.033)  # ~30 FPS


def draw_trajectory_path(img, trajectory, colors, thickness=2):
    """
    Draw trajectory path on image.
    
    Args:
        img: Image to draw on
        trajectory: Deque of (x, y) points
        colors: List of BGR colors
        thickness: Line thickness
    """
    try:
        if len(trajectory) < 2:
            return
        
        trajectory = list(trajectory)
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
