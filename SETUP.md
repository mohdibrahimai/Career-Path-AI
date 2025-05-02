# Career Vision AI Setup Guide

This guide will help you set up the Career Vision AI application on your local machine or server.

## Prerequisites

- Python 3.10 or higher
- PostgreSQL database
- Git (for cloning the repository)

## Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/career-vision-ai.git
cd career-vision-ai
```

2. **Install dependencies**

Using pip:
```bash
# Install PyTorch
pip install torch

# Install other dependencies
pip install flask flask-login flask-sqlalchemy gunicorn psycopg2-binary numpy werkzeug email-validator
```

3. **Set up environment variables**

Create a `.env` file in the root directory with the following variables:
```
DATABASE_URL=postgresql://username:password@localhost:5432/career_vision_ai
SESSION_SECRET=your-secret-key
```

4. **Initialize the database**

Start a Python shell and run:
```python
from app import app, db
with app.app_context():
    db.create_all()
```

## Running the Application

### Development Mode

```bash
python main.py
```

### Production Mode

```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

## Project Structure

- `app.py`: Main Flask application with routes and database configuration
- `career_recommender.py`: Text analysis and embedding generation
- `dqn_agent.py`: Deep Q-Network implementation
- `models.py`: Database models
- `utils.py`: Helper utilities including career lists and information
- `templates/`: HTML templates
- `static/`: CSS and JavaScript files

## Troubleshooting

### Database Connection Issues

If you encounter database connection problems:

1. Verify PostgreSQL is running
2. Check your DATABASE_URL environment variable
3. Make sure the database exists and has the correct permissions

### Model Training

The DQN model will start with random recommendations and improve as users provide feedback. Initial recommendations may not be optimal until sufficient feedback data is collected.

## Contact

For questions or support, please open an issue on the GitHub repository.