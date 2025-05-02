from app import db
from datetime import datetime

class UserFeedback(db.Model):
    """Model for storing user feedback on career recommendations"""
    id = db.Column(db.Integer, primary_key=True)
    resume_text = db.Column(db.Text, nullable=False)
    recommended_career = db.Column(db.String(100), nullable=False)
    rating = db.Column(db.Integer, nullable=False)  # 1-5 rating
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserFeedback {self.id}: {self.recommended_career}, Rating: {self.rating}>'

class ModelState(db.Model):
    """Model for storing DQN model states"""
    id = db.Column(db.Integer, primary_key=True)
    model_data = db.Column(db.LargeBinary, nullable=False)  # Serialized model
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    epsilon = db.Column(db.Float, nullable=False)  # Current epsilon value
    loss = db.Column(db.Float, nullable=True)  # Last training loss
    
    def __repr__(self):
        return f'<ModelState {self.id}: {self.timestamp}, Epsilon: {self.epsilon}>'
