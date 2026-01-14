# Changelog

All notable changes to PosePro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-01-13

### Changed
- **Major refactor**: Split monolithic `app.py` into modular components
  - `config.py` - Configuration constants and thresholds
  - `pose_analyzer.py` - Core pose analysis and scoring logic
  - `chatbot.py` - AI form assistant with keyword-based responses
  - `models.py` - Data models (RepData, SessionData)
  - `camera_handler.py` - Camera capture and video streaming
- Improved code organization following separation of concerns
- Fixed duplicate `time.sleep()` call in mesh frame generation
- Removed empty `src/` directory
- Updated README.md with accurate project structure
- Updated CONTRIBUTING.md with new module architecture

### Technical
- Reduced `app.py` from 1488 lines to ~750 lines
- Centralized configuration in `config.py`
- Introduced `AngleSmoother` class for cleaner angle processing
- Improved `ChatHistory` class for better chat management
- Enhanced `CameraHandler` class with encapsulated camera state

## [2.0.0] - 2025-01-08

### Added
- Real-time pose detection using MediaPipe
- Automatic rep counting with phase detection (up → peak → down)
- Comprehensive scoring system (ROM, symmetry, stability, smoothness, elbow position)
- Performance dashboard with analytics and charts
- Session history and detailed rep metrics
- Personal records tracking
- Leaderboard system with multi-user support
- User profiles with customization options
- AI form assistant chat interface
- Video upload analysis (beta)
- Achievement/badge system
- Background blur using MediaPipe selfie segmentation
- Wrist trajectory visualization
- Cross-platform support (Windows, macOS, Linux)

### Technical
- Flask web framework with SQLite database
- MediaPipe Pose for landmark detection
- OpenCV for video capture and processing
- Responsive modern UI with dark theme

## [1.0.0] - Initial Release

### Added
- Basic pose detection
- Simple rep counting
- Form scoring
