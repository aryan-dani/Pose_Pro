"""
Backend package for PosePro
Contains all backend modules: config, models, pose analyzer, database, chatbot, camera handler
"""

from .config import (
    THRESHOLDS, COLORS, MEDIAPIPE_SETTINGS, 
    SERVER_HOST, SERVER_PORT, SECRET_KEY,
    SCORING_WEIGHTS, GRADE_THRESHOLDS, CAMERA_SETTINGS,
    TRAJECTORY_BUFFER_SIZE
)
from .models import RepData, SessionData
from .pose_analyzer import (
    calculate_shoulder_abduction_from_vertical,
    calculate_torso_tilt_from_vertical,
    calculate_elbow_extension_angle,
    calculate_rep_scores,
    AngleSmoother
)
from .chatbot import get_chatbot_response, chat_history, ChatHistory
from .camera_handler import CameraHandler, draw_trajectory_path, debug_print
from .database import (
    init_database, create_session, save_rep, complete_session,
    get_session_history, get_session_reps, check_personal_records,
    get_overall_stats, get_weekly_progress, get_recent_sessions, get_personal_records,
    get_week_stats, get_month_stats, get_score_trend,
    get_user_profile, update_user_profile, create_user,
    get_leaderboard, get_user_rank, get_user_achievements
)
