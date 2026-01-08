<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/Flask-2.0+-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/MediaPipe-Latest-orange.svg" alt="MediaPipe">
  <img src="https://img.shields.io/badge/OpenCV-4.0+-red.svg" alt="OpenCV">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</p>

<h1 align="center">PosePro</h1>

<p align="center">
  <strong>AI-Powered Real-Time Shoulder Raise Form Analysis</strong>
</p>

<p align="center">
  Perfect your lateral raise form with instant AI feedback, detailed metrics, and progress tracking.
</p>

---

## Features

### Real-Time Form Analysis
- **Live Camera Tracking** - Pose detection using MediaPipe with background blur
- **Automatic Rep Counting** - Intelligent phase detection (up → peak → down)
- **Instant Feedback** - Real-time angle visualization and trajectory tracking

### Comprehensive Metrics
| Metric | Description |
|--------|-------------|
| **Range of Motion** | Measures arm elevation angle (ideal: 70-90°) |
| **Bilateral Symmetry** | Compares left vs right arm angles |
| **Torso Stability** | Detects body lean or sway |
| **Elbow Position** | Monitors elbow angle (ideal: 160-180°) |
| **Rep Duration** | Tracks timing and smoothness |

### Progress Tracking
- **Performance Dashboard** - Charts, trends, and analytics
- **Session History** - Review past workouts and rep details
- **Personal Records** - Track your bests for score and ROM
- **Leaderboard** - Compete with others (multi-user support)
- **Achievements** - Unlock badges as you progress

### AI Form Assistant
- Chat interface for technique advice and form corrections
- Personalized recommendations based on your performance

---

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Pose_Pro.git
   cd Pose_Pro
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   ```
   http://localhost:5000
   ```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `flask` | Web framework |
| `opencv-python` | Video capture and image processing |
| `mediapipe` | Pose detection and landmark tracking |
| `numpy` | Numerical computations |

### Install all dependencies:
```bash
pip install flask opencv-python mediapipe numpy
```

---

## How to Use

### Starting a Workout Session

1. **Position yourself** - Stand 2-3 meters from your camera with good lighting
2. **Navigate to Live Analysis** - Click "Start Training" or go to `/camera`
3. **Start tracking** - Click the "Start Tracking" button
4. **Perform lateral raises** - The AI will automatically detect and score each rep
5. **Review your results** - Check your dashboard for detailed analytics

### Understanding Your Score

Each rep is graded on a 100-point scale:

| Grade | Score Range | Meaning |
|-------|-------------|---------|
| A+ | 90-100 | Excellent form |
| A/A- | 80-89 | Great form |
| B+/B/B- | 65-79 | Good form, room for improvement |
| C+/C | 55-64 | Needs work |
| F | <55 | Poor form |

### Form Tips
- Keep your **torso stable** - avoid leaning
- Maintain **slight elbow bend** (160-180°)
- Raise arms to **shoulder height** (70-90°)
- Move **smoothly** - avoid jerky motions
- Keep both arms **symmetric**

---

## Project Structure

```
Pose_Pro/
├── app.py              # Main Flask application & pose analysis logic
├── database.py         # SQLite database operations
├── static/
│   └── css/
│       └── style.css   # Application styling
├── templates/
│   ├── index.html      # Home page
│   ├── camera.html     # Live analysis view
│   ├── dashboard.html  # Analytics dashboard
│   ├── history.html    # Session history
│   ├── leaderboard.html# Competitive rankings
│   ├── profile.html    # User profile
│   ├── chat.html       # AI form assistant
│   └── upload.html     # Video upload
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
├── LICENSE             # MIT License
├── README.md           # This file
└── CONTRIBUTING.md     # Contribution guidelines
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | Random | Flask session secret key |
| `TF_CPP_MIN_LOG_LEVEL` | 3 | TensorFlow logging level |

### Detection Thresholds

Thresholds can be adjusted in `app.py`:

```python
THRESHOLDS = {
    'rep_start': 15.0,        # Angle to start rep
    'rep_peak_min': 40.0,     # Minimum peak angle
    'rep_end': 12.0,          # Angle to end rep
    'ideal_rom_min': 50.0,    # Ideal ROM minimum
    'ideal_rom_max': 90.0,    # Ideal ROM maximum
    # ... more in app.py
}
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/camera` | GET | Live analysis page |
| `/dashboard` | GET | Analytics dashboard |
| `/api/status` | GET | Current tracking status |
| `/api/start_tracking` | POST | Start rep tracking |
| `/api/stop_tracking` | POST | Stop tracking |
| `/api/dashboard/stats` | GET | Dashboard statistics |
| `/api/history` | GET | Session history |

---

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Steps
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Pose detection framework by Google
- [OpenCV](https://opencv.org/) - Computer vision library
- [Flask](https://flask.palletsprojects.com/) - Python web framework

---

## Support

Having issues? Here's how to get help:

1. Check the [Issues](https://github.com/yourusername/Pose_Pro/issues) page
2. Create a new issue with:
   - Your OS and Python version
   - Steps to reproduce the problem
   - Expected vs actual behavior
   - Any error messages
