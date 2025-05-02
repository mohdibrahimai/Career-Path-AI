# 🚀 Career Vision AI — Intelligent, Self-Learning Career Recommendation System

Career Vision AI is a smart web application that recommends personalized career paths using Reinforcement Learning (DQN). It's designed to help users—students, professionals, or career switchers—find the most suitable career based on their resume or skill profile.

## 🔍 Key Features

### 📄 Smart Resume Analysis
Uses a custom keyword-based parsing engine to extract insights from your resume: skills, tools, technologies, and experience indicators.

### 🧠 AI-Powered Career Recommendations
Built with a Deep Q-Learning Agent (DQN) using PyTorch, the model selects the best-fit career from over 25+ professions.

### 🔄 Feedback-Driven Learning Loop
Each time a user rates a recommendation (1–5), the agent uses this as a reward signal to improve its future predictions.

### 📊 Learning Analytics Dashboard
Track how well the model is performing with metrics like average rating per career, reward history, and decision trends.

## ⚙️ How It Works

1. **Input → Embedding**  
   User provides a resume or skills description, which is transformed into a vector representation using a custom NLP model.

2. **AI Agent → Career Decision**  
   The DQN model takes this embedding and chooses the most suitable career path.

3. **User Feedback → Model Update**  
   You rate the recommendation, and the model learns from this using reinforcement learning principles (state → action → reward).

4. **Self-Improving Intelligence**  
   Over time, the system adapts and becomes smarter with every user interaction.

## 🧱 Technical Architecture

- **Frontend:** Responsive Bootstrap-based UI
- **Backend:** Python Flask application
- **Database:** PostgreSQL (stores user feedback, model experience)
- **AI Core:** Custom PyTorch-based Deep Q-Network
- **Text Analysis:** Keyword-based embedding engine (lightweight & fast)

## 🧪 Real-World Impact

This project shows how Reinforcement Learning can go beyond games and simulations—into the real world—by helping people make informed career decisions through intelligent, adaptive recommendations.

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies with `pip install -r requirements.txt`
3. Set up a PostgreSQL database and configure DATABASE_URL
4. Run the application with `python main.py` or `gunicorn --bind 0.0.0.0:5000 main:app`

## 📸 Screenshots

(Coming soon)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).