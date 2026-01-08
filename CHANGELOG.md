# Changelog

All notable changes to PosePro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
