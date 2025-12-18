"""
Database module for Shoulder Raise Analyzer
SQLite database for storing workout sessions, rep scores, and user progress
"""

import sqlite3
import datetime
import json
import os
import hashlib
from contextlib import contextmanager
from typing import List, Dict, Optional, Any

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'shoulder_analyzer.db')

def init_database():
    """Initialize the database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Users table - create basic table first
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add new columns if they don't exist (migration for existing databases)
        new_columns = [
            ('display_name', 'TEXT'),
            ('email', 'TEXT'),
            ('avatar_color', "TEXT DEFAULT '#10b981'"),
            ('bio', 'TEXT'),
            ('fitness_goal', "TEXT DEFAULT 'general'"),
            ('experience_level', "TEXT DEFAULT 'beginner'"),
            ('last_active', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
        ]
        
        for col_name, col_type in new_columns:
            try:
                cursor.execute(f'ALTER TABLE users ADD COLUMN {col_name} {col_type}')
            except sqlite3.OperationalError:
                pass  # Column already exists
        
        # Workout sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workout_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER DEFAULT 1,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                total_reps INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0,
                best_score REAL DEFAULT 0,
                best_grade TEXT DEFAULT 'N/A',
                total_duration_seconds REAL DEFAULT 0,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Individual reps table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                rep_number INTEGER NOT NULL,
                score REAL NOT NULL,
                grade TEXT NOT NULL,
                rom_total REAL,
                rom_peak REAL,
                symmetry_diff REAL,
                torso_stability REAL,
                elbow_angle REAL,
                duration_seconds REAL,
                warnings TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES workout_sessions(id)
            )
        ''')
        
        # Personal records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER DEFAULT 1,
                record_type TEXT NOT NULL,
                record_value REAL NOT NULL,
                rep_id INTEGER,
                session_id INTEGER,
                achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (rep_id) REFERENCES reps(id),
                FOREIGN KEY (session_id) REFERENCES workout_sessions(id)
            )
        ''')
        
        # Daily stats summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER DEFAULT 1,
                date DATE NOT NULL,
                total_sessions INTEGER DEFAULT 0,
                total_reps INTEGER DEFAULT 0,
                avg_score REAL DEFAULT 0,
                best_score REAL DEFAULT 0,
                total_workout_time_seconds REAL DEFAULT 0,
                UNIQUE(user_id, date),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Leaderboard table for competitive rankings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                score_type TEXT NOT NULL,
                score_value REAL NOT NULL,
                achieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Create default user
        cursor.execute('''
            INSERT OR IGNORE INTO users (id, username, display_name) VALUES (1, 'default_user', 'Athlete')
        ''')
        
        conn.commit()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Session Management
def create_session(user_id: int = 1) -> Optional[int]:
    """Create a new workout session and return its ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO workout_sessions (user_id, start_time)
            VALUES (?, ?)
        ''', (user_id, datetime.datetime.now()))
        conn.commit()
        return cursor.lastrowid

def complete_session(session_id: int, total_reps: int, avg_score: float, best_grade: str):
    """Complete a workout session with summary stats"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get session start time to calculate duration
        cursor.execute('SELECT start_time FROM workout_sessions WHERE id = ?', (session_id,))
        row = cursor.fetchone()
        start_time = datetime.datetime.fromisoformat(row['start_time']) if row else datetime.datetime.now()
        duration = (datetime.datetime.now() - start_time).total_seconds()
        
        # Get best score from reps
        cursor.execute('SELECT MAX(score) as best FROM reps WHERE session_id = ?', (session_id,))
        best_row = cursor.fetchone()
        best_score = best_row['best'] if best_row and best_row['best'] else avg_score
        
        cursor.execute('''
            UPDATE workout_sessions 
            SET end_time = ?, total_reps = ?, avg_score = ?, 
                best_score = ?, best_grade = ?, total_duration_seconds = ?
            WHERE id = ?
        ''', (datetime.datetime.now(), total_reps, avg_score, 
              best_score, best_grade, duration, session_id))
        conn.commit()
        
        # Update daily stats
        update_daily_stats(1, total_reps, avg_score, best_score, duration)

def save_rep(session_id: int, rep_number: int, score: float, grade: str,
             rom_angle: float, symmetry_diff: float, torso_stability: float,
             elbow_angle: float, duration: float, warnings: list) -> Optional[int]:
    """Save a single rep to the database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        warnings_json = json.dumps(warnings if warnings else [])
        
        cursor.execute('''
            INSERT INTO reps (
                session_id, rep_number, score, grade, rom_total,
                symmetry_diff, torso_stability, elbow_angle, 
                duration_seconds, warnings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, rep_number, score, grade, rom_angle,
            symmetry_diff, torso_stability, elbow_angle, duration, warnings_json
        ))
        conn.commit()
        
        return cursor.lastrowid

def check_personal_records(user_id: int, score: float, rom: float):
    """Check and update personal records"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        records_to_check = [
            ('best_score', score),
            ('best_rom', rom),
        ]
        
        for record_type, value in records_to_check:
            cursor.execute('''
                SELECT record_value FROM personal_records 
                WHERE user_id = ? AND record_type = ?
                ORDER BY record_value DESC LIMIT 1
            ''', (user_id, record_type))
            
            result = cursor.fetchone()
            
            if result is None or value > result['record_value']:
                cursor.execute('''
                    INSERT INTO personal_records 
                    (user_id, record_type, record_value)
                    VALUES (?, ?, ?)
                ''', (user_id, record_type, value))
        
        conn.commit()

def update_daily_stats(user_id: int, reps: int, avg_score: float, 
                       best_score: float, duration: float):
    """Update or create daily stats entry"""
    today = datetime.date.today()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM daily_stats WHERE user_id = ? AND date = ?
        ''', (user_id, today))
        
        existing = cursor.fetchone()
        
        if existing:
            new_total_reps = existing['total_reps'] + reps
            new_sessions = existing['total_sessions'] + 1
            new_avg = (existing['avg_score'] * existing['total_sessions'] + avg_score) / new_sessions
            new_best = max(existing['best_score'], best_score)
            new_time = existing['total_workout_time_seconds'] + duration
            
            cursor.execute('''
                UPDATE daily_stats 
                SET total_sessions = ?, total_reps = ?, avg_score = ?,
                    best_score = ?, total_workout_time_seconds = ?
                WHERE user_id = ? AND date = ?
            ''', (new_sessions, new_total_reps, new_avg, new_best, new_time, user_id, today))
        else:
            cursor.execute('''
                INSERT INTO daily_stats 
                (user_id, date, total_sessions, total_reps, avg_score, 
                 best_score, total_workout_time_seconds)
                VALUES (?, ?, 1, ?, ?, ?, ?)
            ''', (user_id, today, reps, avg_score, best_score, duration))
        
        conn.commit()

# Data Retrieval
def get_session_history(user_id: int = 1, limit: int = 50, offset: int = 0) -> List[Dict]:
    """Get workout session history"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM workout_sessions 
            WHERE user_id = ? 
            ORDER BY start_time DESC 
            LIMIT ? OFFSET ?
        ''', (user_id, limit, offset))
        
        sessions = []
        for row in cursor.fetchall():
            session = dict(row)
            # Format dates for JSON
            if session.get('start_time'):
                session['start_time'] = str(session['start_time'])
            if session.get('end_time'):
                session['end_time'] = str(session['end_time'])
            if session.get('created_at'):
                session['created_at'] = str(session['created_at'])
            sessions.append(session)
        
        return sessions

def get_session_reps(session_id: int) -> List[Dict]:
    """Get all reps for a specific session"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM reps 
            WHERE session_id = ? 
            ORDER BY rep_number ASC
        ''', (session_id,))
        
        reps = []
        for row in cursor.fetchall():
            rep = dict(row)
            # Parse warnings JSON
            if rep.get('warnings'):
                try:
                    rep['warnings'] = json.loads(rep['warnings'])
                except:
                    rep['warnings'] = []
            if rep.get('timestamp'):
                rep['timestamp'] = str(rep['timestamp'])
            reps.append(rep)
        
        return reps

def get_personal_records(user_id: int = 1) -> List[Dict]:
    """Get all personal records for a user"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT record_type, MAX(record_value) as value, 
                   MAX(achieved_at) as achieved_at
            FROM personal_records 
            WHERE user_id = ?
            GROUP BY record_type
        ''', (user_id,))
        
        records = []
        for row in cursor.fetchall():
            records.append({
                'type': row['record_type'],
                'value': row['value'],
                'achieved_at': str(row['achieved_at']) if row['achieved_at'] else None
            })
        
        return records

def get_recent_sessions(user_id: int = 1, limit: int = 10) -> List[Dict]:
    """Get recent sessions for dashboard"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, start_time, total_reps, avg_score, best_grade, 
                   total_duration_seconds
            FROM workout_sessions 
            WHERE user_id = ? AND end_time IS NOT NULL
            ORDER BY start_time DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        sessions = []
        for row in cursor.fetchall():
            session = dict(row)
            if session.get('start_time'):
                session['start_time'] = str(session['start_time'])
            session['duration_formatted'] = format_duration(session.get('total_duration_seconds', 0) or 0)
            sessions.append(session)
        
        return sessions

def get_daily_stats(user_id: int = 1, days: int = 30) -> List[Dict]:
    """Get daily stats for the last N days"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM daily_stats 
            WHERE user_id = ? 
            ORDER BY date DESC 
            LIMIT ?
        ''', (user_id, days))
        
        stats = []
        for row in cursor.fetchall():
            stat = dict(row)
            if stat.get('date'):
                stat['date'] = str(stat['date'])
            stats.append(stat)
        
        return stats

def get_overall_stats(user_id: int = 1) -> Dict[str, Any]:
    """Get overall statistics for a user"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Total stats from sessions
        cursor.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                COALESCE(SUM(total_reps), 0) as total_reps,
                COALESCE(AVG(avg_score), 0) as avg_score,
                COALESCE(MAX(best_score), 0) as best_score,
                COALESCE(SUM(total_duration_seconds), 0) as total_workout_time
            FROM workout_sessions 
            WHERE user_id = ? AND end_time IS NOT NULL
        ''', (user_id,))
        
        row = cursor.fetchone()
        session_stats = {
            'total_sessions': row['total_sessions'] or 0,
            'total_reps': row['total_reps'] or 0,
            'avg_score': row['avg_score'] or 0,
            'best_score': row['best_score'] or 0,
            'total_workout_time': row['total_workout_time'] or 0
        }
        
        # Grade distribution
        cursor.execute('''
            SELECT grade, COUNT(*) as count
            FROM reps r
            JOIN workout_sessions s ON r.session_id = s.id
            WHERE s.user_id = ?
            GROUP BY grade
        ''', (user_id,))
        
        grade_dist = {row['grade']: row['count'] for row in cursor.fetchall()}
        
        return {
            **session_stats,
            'grade_distribution': grade_dist,
            'total_workout_time_formatted': format_duration(session_stats.get('total_workout_time', 0))
        }

def get_weekly_progress(user_id: int = 1) -> List[Dict]:
    """Get weekly progress for charts"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                date,
                total_reps,
                avg_score,
                best_score,
                total_workout_time_seconds as total_time
            FROM daily_stats 
            WHERE user_id = ?
            ORDER BY date DESC
            LIMIT 30
        ''', (user_id,))
        
        progress = []
        for row in cursor.fetchall():
            progress.append({
                'date': str(row['date']),
                'total_reps': row['total_reps'] or 0,
                'avg_score': row['avg_score'] or 0,
                'best_score': row['best_score'] or 0,
                'total_time': row['total_time'] or 0
            })
        
        return progress

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if not seconds:
        return "0m"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"

def get_week_stats(user_id: int = 1) -> Dict[str, Any]:
    """Get statistics for the current week"""
    today = datetime.date.today()
    week_start = today - datetime.timedelta(days=today.weekday())
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get week's sessions and reps
        cursor.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                COALESCE(SUM(total_reps), 0) as total_reps,
                COALESCE(AVG(avg_score), 0) as avg_score,
                COALESCE(MAX(best_score), 0) as best_score,
                COALESCE(SUM(total_duration_seconds), 0) as total_time
            FROM workout_sessions 
            WHERE user_id = ? 
            AND date(start_time) >= ?
            AND end_time IS NOT NULL
        ''', (user_id, week_start))
        
        row = cursor.fetchone()
        
        return {
            'total_sessions': row['total_sessions'] or 0,
            'total_reps': row['total_reps'] or 0,
            'avg_score': row['avg_score'] or 0,
            'best_score': row['best_score'] or 0,
            'total_time': row['total_time'] or 0
        }

def get_month_stats(user_id: int = 1) -> Dict[str, Any]:
    """Get statistics for the current month"""
    today = datetime.date.today()
    month_start = today.replace(day=1)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get month's sessions and reps
        cursor.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                COALESCE(SUM(total_reps), 0) as total_reps,
                COALESCE(AVG(avg_score), 0) as avg_score,
                COALESCE(MAX(best_score), 0) as best_score,
                COALESCE(SUM(total_duration_seconds), 0) as total_time
            FROM workout_sessions 
            WHERE user_id = ? 
            AND date(start_time) >= ?
            AND end_time IS NOT NULL
        ''', (user_id, month_start))
        
        row = cursor.fetchone()
        
        return {
            'total_sessions': row['total_sessions'] or 0,
            'total_reps': row['total_reps'] or 0,
            'avg_score': row['avg_score'] or 0,
            'best_score': row['best_score'] or 0,
            'total_time': row['total_time'] or 0
        }

def get_score_trend(user_id: int = 1) -> float:
    """Calculate score trend comparing this week vs last week"""
    today = datetime.date.today()
    this_week_start = today - datetime.timedelta(days=today.weekday())
    last_week_start = this_week_start - datetime.timedelta(days=7)
    last_week_end = this_week_start - datetime.timedelta(days=1)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # This week's average
        cursor.execute('''
            SELECT COALESCE(AVG(avg_score), 0) as avg_score
            FROM workout_sessions 
            WHERE user_id = ? 
            AND date(start_time) >= ?
            AND end_time IS NOT NULL
        ''', (user_id, this_week_start))
        
        this_week_row = cursor.fetchone()
        this_week_avg = this_week_row['avg_score'] or 0
        
        # Last week's average
        cursor.execute('''
            SELECT COALESCE(AVG(avg_score), 0) as avg_score
            FROM workout_sessions 
            WHERE user_id = ? 
            AND date(start_time) >= ?
            AND date(start_time) <= ?
            AND end_time IS NOT NULL
        ''', (user_id, last_week_start, last_week_end))
        
        last_week_row = cursor.fetchone()
        last_week_avg = last_week_row['avg_score'] or 0
        
        # Calculate trend (difference)
        if last_week_avg == 0:
            return 0.0
        
        return this_week_avg - last_week_avg

# Initialize database on module import
init_database()

# ==================== USER PROFILE FUNCTIONS ====================

def get_user_profile(user_id: int = 1) -> Optional[Dict]:
    """Get user profile information"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, username, display_name, email, avatar_color, bio, 
                   fitness_goal, experience_level, created_at, last_active
            FROM users WHERE id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'id': row['id'],
                'username': row['username'],
                'display_name': row['display_name'] or row['username'],
                'email': row['email'],
                'avatar_color': row['avatar_color'] or '#10b981',
                'bio': row['bio'],
                'fitness_goal': row['fitness_goal'] or 'general',
                'experience_level': row['experience_level'] or 'beginner',
                'created_at': str(row['created_at']) if row['created_at'] else None,
                'last_active': str(row['last_active']) if row['last_active'] else None
            }
        return None

def update_user_profile(user_id: int, **kwargs) -> bool:
    """Update user profile with provided fields"""
    allowed_fields = ['display_name', 'email', 'avatar_color', 'bio', 'fitness_goal', 'experience_level']
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    
    if not updates:
        return False
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [user_id]
        
        cursor.execute(f'''
            UPDATE users SET {set_clause}, last_active = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', values)
        
        conn.commit()
        return cursor.rowcount > 0

def create_user(username: str, display_name: Optional[str] = None, email: Optional[str] = None) -> Optional[int]:
    """Create a new user and return user ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO users (username, display_name, email)
                VALUES (?, ?, ?)
            ''', (username, display_name or username, email))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None

# ==================== LEADERBOARD FUNCTIONS ====================

def get_leaderboard(score_type: str = 'best_score', limit: int = 10) -> List[Dict]:
    """Get leaderboard rankings"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get best scores from workout sessions for each user
        if score_type == 'best_score':
            cursor.execute('''
                SELECT u.id as user_id, u.display_name, u.avatar_color,
                       MAX(ws.best_score) as score,
                       COUNT(DISTINCT ws.id) as total_sessions,
                       SUM(ws.total_reps) as total_reps
                FROM users u
                LEFT JOIN workout_sessions ws ON u.id = ws.user_id AND ws.end_time IS NOT NULL
                GROUP BY u.id
                HAVING score IS NOT NULL
                ORDER BY score DESC
                LIMIT ?
            ''', (limit,))
        elif score_type == 'total_reps':
            cursor.execute('''
                SELECT u.id as user_id, u.display_name, u.avatar_color,
                       COALESCE(SUM(ws.total_reps), 0) as score,
                       COUNT(DISTINCT ws.id) as total_sessions,
                       SUM(ws.total_reps) as total_reps
                FROM users u
                LEFT JOIN workout_sessions ws ON u.id = ws.user_id AND ws.end_time IS NOT NULL
                GROUP BY u.id
                HAVING score > 0
                ORDER BY score DESC
                LIMIT ?
            ''', (limit,))
        elif score_type == 'avg_score':
            cursor.execute('''
                SELECT u.id as user_id, u.display_name, u.avatar_color,
                       AVG(ws.avg_score) as score,
                       COUNT(DISTINCT ws.id) as total_sessions,
                       SUM(ws.total_reps) as total_reps
                FROM users u
                LEFT JOIN workout_sessions ws ON u.id = ws.user_id AND ws.end_time IS NOT NULL
                GROUP BY u.id
                HAVING score IS NOT NULL AND total_sessions >= 3
                ORDER BY score DESC
                LIMIT ?
            ''', (limit,))
        else:
            return []
        
        results = []
        rank = 1
        for row in cursor.fetchall():
            results.append({
                'rank': rank,
                'user_id': row['user_id'],
                'display_name': row['display_name'] or 'Athlete',
                'avatar_color': row['avatar_color'] or '#10b981',
                'score': row['score'],
                'total_sessions': row['total_sessions'] or 0,
                'total_reps': row['total_reps'] or 0
            })
            rank += 1
        
        return results

def get_user_rank(user_id: int, score_type: str = 'best_score') -> Dict:
    """Get a specific user's rank and stats"""
    leaderboard = get_leaderboard(score_type, limit=100)
    
    for entry in leaderboard:
        if entry['user_id'] == user_id:
            return {
                'rank': entry['rank'],
                'total_users': len(leaderboard),
                'score': entry['score'],
                'percentile': round((1 - (entry['rank'] - 1) / max(len(leaderboard), 1)) * 100, 1)
            }
    
    return {'rank': None, 'total_users': len(leaderboard), 'score': 0, 'percentile': 0}

def get_user_achievements(user_id: int) -> List[Dict]:
    """Get user achievements based on their stats"""
    achievements = []
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get user stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                COALESCE(SUM(total_reps), 0) as total_reps,
                COALESCE(MAX(best_score), 0) as best_score,
                COALESCE(AVG(avg_score), 0) as avg_score
            FROM workout_sessions 
            WHERE user_id = ? AND end_time IS NOT NULL
        ''', (user_id,))
        
        stats = cursor.fetchone()
        
        # Define achievements
        if stats['total_sessions'] >= 1:
            achievements.append({'id': 'first_workout', 'name': 'First Steps', 'icon': '🎯', 'description': 'Complete your first workout'})
        if stats['total_sessions'] >= 5:
            achievements.append({'id': 'five_sessions', 'name': 'Getting Started', 'icon': '🌟', 'description': 'Complete 5 workouts'})
        if stats['total_sessions'] >= 10:
            achievements.append({'id': 'ten_sessions', 'name': 'Dedicated', 'icon': '💪', 'description': 'Complete 10 workouts'})
        if stats['total_sessions'] >= 25:
            achievements.append({'id': 'committed', 'name': 'Committed', 'icon': '🔥', 'description': 'Complete 25 workouts'})
        if stats['total_reps'] >= 50:
            achievements.append({'id': 'fifty_reps', 'name': 'Rep Machine', 'icon': '⚡', 'description': 'Complete 50 total reps'})
        if stats['total_reps'] >= 100:
            achievements.append({'id': 'hundred_reps', 'name': 'Century Club', 'icon': '🏆', 'description': 'Complete 100 total reps'})
        if stats['best_score'] >= 85:
            achievements.append({'id': 'high_scorer', 'name': 'High Scorer', 'icon': '🎖️', 'description': 'Score 85+ on a rep'})
        if stats['best_score'] >= 95:
            achievements.append({'id': 'perfect_form', 'name': 'Perfect Form', 'icon': '👑', 'description': 'Score 95+ on a rep'})
        if stats['avg_score'] >= 75:
            achievements.append({'id': 'consistent', 'name': 'Consistent', 'icon': '📈', 'description': 'Maintain 75+ average score'})
        
        return achievements
