# Python Development Environment

This is a Python development environment with the following features:

- Virtual environment management
- Code formatting with Black
- Linting with Flake8
- Testing with pytest
- Environment variable management with python-dotenv

## Setup Instructions

1. Create and activate the virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/MacOS
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Tools

- **Black**: Code formatter
  ```bash
  black .
  ```

- **Flake8**: Linter
  ```bash
  flake8 .
  ```

- **pytest**: Testing framework
  ```bash
  pytest
  ```

## Project Structure

```
.
├── venv/                  # Virtual environment
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

Add your project-specific files and directories as needed. 