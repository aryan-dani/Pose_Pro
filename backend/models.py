"""
Data Models module for PosePro
Contains data classes for sessions and reps
"""

import datetime


class RepData:
    """Stores data for a single rep including metrics and scores."""
    
    def __init__(self):
        self.rep_number = 0
        self.start_time = 0.0
        self.end_time = 0.0
        self.peak_time = 0.0
        self.rep_duration = 0.0
        
        # Angle data
        self.left_angles = []
        self.right_angles = []
        self.avg_angles = []
        self.timestamps = []
        
        # Stability data
        self.torso_tilts = []
        self.shoulder_elevations = []
        
        # Elbow data
        self.elbow_angles_left = []
        self.elbow_angles_right = []
        
        # Trajectory data
        self.wrist_positions = []
        self.left_wrist_trajectory = []
        self.right_wrist_trajectory = []
        
        # Calculated metrics
        self.rom_peak_angle = 0.0
        self.rom_min_angle = 0.0
        self.rom_total = 0.0
        self.symmetry_diff = 0.0
        self.torso_lean_max = 0.0
        self.smoothness_index = 0.0
        self.elbow_angle_avg = 0.0
        self.time_between_reps = 0.0
        
        # Scores (0.0 to 1.0 for components, 0-100 for overall)
        self.score_rom = 0.0
        self.score_symmetry = 0.0
        self.score_torso_stability = 0.0
        self.score_smoothness = 0.0
        self.score_elbow_position = 0.0
        self.score_overall = 0.0
        self.form_grade = "C"
        self.warnings = []


class SessionData:
    """Stores data for a workout session."""
    
    def __init__(self):
        self.start_time = datetime.datetime.now()
        self.reps = []
        self.frame_data = []
        self.baseline_hip_y = 0.0
        self.rep_durations = []
        self.last_rep_end_time = 0.0
    
    def reset(self):
        """Reset session data for a new session."""
        self.__init__()
    
    def get_summary(self):
        """
        Get session summary statistics.
        
        Returns:
            dict: Session statistics
        """
        if not self.reps:
            return None
        
        import numpy as np
        
        scores = [rep.score_overall for rep in self.reps]
        rom_values = [rep.rom_total for rep in self.reps]
        symmetry_values = [rep.symmetry_diff for rep in self.reps]
        durations = [rep.rep_duration for rep in self.reps]
        
        return {
            'total_reps': len(self.reps),
            'avg_score': float(np.mean(scores)),
            'best_score': float(max(scores)),
            'worst_score': float(min(scores)),
            'avg_rom': float(np.mean(rom_values)),
            'avg_symmetry': float(np.mean(symmetry_values)),
            'avg_duration': float(np.mean(durations)),
            'best_grade': self.reps[np.argmax(scores)].form_grade,
            'session_duration': (datetime.datetime.now() - self.start_time).total_seconds()
        }
