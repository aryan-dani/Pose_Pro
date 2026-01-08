# Contributing to PosePro

First off, thank you for considering contributing to PosePro! 🎉

This document provides guidelines and steps for contributing. Following these guidelines helps communicate that you respect the time of the developers managing and developing this open source project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

---

## Code of Conduct

This project and everyone participating in it is governed by our commitment to creating a welcoming and inclusive environment. By participating, you are expected to:

- **Be respectful** - Treat everyone with respect and kindness
- **Be constructive** - Provide helpful feedback and be open to receiving it
- **Be inclusive** - Welcome newcomers and help them get started
- **Be patient** - Remember that everyone has different skill levels

---

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- Python 3.8 or higher
- Git installed
- A GitHub account
- Basic knowledge of Flask and Python

### Fork and Clone

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Pose_Pro.git
   cd Pose_Pro
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/Pose_Pro.git
   ```

---

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**When creating a bug report, include:**

- **Clear title** - Descriptive summary of the issue
- **Steps to reproduce** - Detailed steps to reproduce the behavior
- **Expected behavior** - What you expected to happen
- **Actual behavior** - What actually happened
- **Screenshots** - If applicable
- **Environment details**:
  - OS (Windows/macOS/Linux)
  - Python version
  - Browser (for UI issues)
  - Webcam model (for camera issues)

### 💡 Suggesting Features

Feature suggestions are welcome! Please include:

- **Clear description** - What feature you'd like
- **Use case** - Why this would be useful
- **Possible implementation** - Any ideas on how it could work

### 📝 Improving Documentation

Documentation improvements are always appreciated:

- Fix typos or clarify confusing sections
- Add examples or tutorials
- Improve code comments
- Translate documentation

### 💻 Contributing Code

Look for issues labeled:
- `good first issue` - Great for newcomers
- `help wanted` - We'd love your help
- `enhancement` - New features or improvements
- `bug` - Something isn't working

---

## Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install development dependencies** (if any):
   ```bash
   pip install -r requirements-dev.txt  # If available
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Run tests** (if available):
   ```bash
   python -m pytest tests/
   ```

---

## Pull Request Process

### Before Submitting

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Keep your fork up to date**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

3. **Make your changes**:
   - Write clean, readable code
   - Add comments where necessary
   - Update documentation if needed

4. **Test your changes**:
   - Ensure the application runs without errors
   - Test affected features manually
   - Run any existing tests

### Submitting Your PR

1. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   ```

   **Commit message format**:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation only
   - `style:` - Formatting, no code change
   - `refactor:` - Code restructuring
   - `test:` - Adding tests
   - `chore:` - Maintenance tasks

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Open a Pull Request** on GitHub:
   - Use a clear, descriptive title
   - Reference any related issues (`Fixes #123`)
   - Describe what changes you made and why
   - Include screenshots for UI changes

### Review Process

- A maintainer will review your PR
- Be responsive to feedback and questions
- Make requested changes promptly
- Once approved, your PR will be merged

---

## Style Guidelines

### Python Code Style

- Follow **PEP 8** style guidelines
- Use **meaningful variable names**
- Keep functions **focused and small**
- Add **docstrings** to functions and classes

```python
def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle between three points.
    
    Args:
        point_a: First point coordinates
        point_b: Vertex point coordinates
        point_c: Third point coordinates
    
    Returns:
        float: Angle in degrees
    """
    # Implementation...
```

### HTML/CSS Style

- Use **semantic HTML5** elements
- Keep CSS **organized and commented**
- Use **meaningful class names**
- Follow the existing CSS custom properties pattern

### JavaScript Style

- Use **modern ES6+** syntax
- Add **comments** for complex logic
- Handle **errors** gracefully
- Use **async/await** for asynchronous operations

---

## Project Structure

When adding new features, follow the existing structure:

```
Pose_Pro/
├── app.py              # Main app - add routes and logic here
├── database.py         # Database operations
├── static/css/         # Add styles here
├── templates/          # Add HTML templates here
└── tests/              # Add tests here (if applicable)
```

---

## Community

### Getting Help

- **GitHub Issues** - For bug reports and feature requests
- **Discussions** - For questions and ideas (if enabled)

### Recognition

Contributors will be:
- Listed in the README acknowledgments
- Credited in release notes for significant contributions

---

## Thank You! 🙏

Your contributions make PosePro better for everyone. Whether it's a bug fix, new feature, or documentation improvement - every contribution matters!

Happy coding! 💪
