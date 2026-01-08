# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| 1.x.x   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** create a public GitHub issue
2. Email the maintainers directly with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

We will respond within 48 hours and work with you to address the issue.

## Security Best Practices

When running PosePro:

- Keep dependencies updated (`pip install --upgrade -r requirements.txt`)
- Use environment variables for sensitive configuration
- Run behind a reverse proxy (nginx/Apache) in production
- Enable HTTPS in production environments
