import torch
import numpy as np
import re
from collections import Counter
import logging
import math

logger = logging.getLogger(__name__)

class CareerRecommender:
    """
    A class for recommending careers based on resume text
    using an enhanced hybrid NLP approach with semantic features
    """
    
    def __init__(self):
        """Initialize the career recommender with a vocabulary of keywords and NLP features"""
        logger.info("Initializing enhanced CareerRecommender with NLP features")
        
        # Define skill categories for semantic domain association
        self.skill_categories = {
            "programming": [
                "python", "java", "javascript", "c++", "typescript", "ruby", "php", "golang", "swift", "kotlin",
                "scala", "rust", "perl", "haskell", "fortran", "assembly", "objective-c", "bash", "shell", "powershell"
            ],
            "data_science": [
                "data", "analytics", "statistics", "machine learning", "ai", "artificial intelligence", 
                "deep learning", "nlp", "natural language", "computer vision", "neural networks", "algorithm",
                "regression", "classification", "clustering", "feature engineering", "predictive modeling",
                "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "r language", "matlab"
            ],
            "web_development": [
                "web", "frontend", "backend", "fullstack", "react", "angular", "vue", "node", "express", "django",
                "flask", "html", "css", "bootstrap", "jquery", "rest", "api", "responsive", "spa", "progressive",
                "webpack", "babel", "redux", "graphql", "websocket", "scss", "sass", "tailwind", "bulma"
            ],
            "devops": [
                "devops", "ci/cd", "jenkins", "docker", "kubernetes", "aws", "azure", "gcp", "cloud",
                "terraform", "ansible", "puppet", "chef", "prometheus", "grafana", "elk", "logging", "monitoring",
                "infrastructure", "microservices", "serverless", "lambda", "git", "github actions", "gitlab ci"
            ],
            "databases": [
                "sql", "nosql", "mysql", "postgresql", "mongodb", "database", "data modeling", "etl",
                "oracle", "sqlite", "redis", "cassandra", "neo4j", "couchdb", "dynamodb", "mariadb", "indexing",
                "query optimization", "acid", "transactions", "sharding", "replication", "backup", "recovery"
            ],
            "mobile": [
                "mobile", "android", "ios", "react native", "flutter", "xamarin", "swift", "kotlin", "objective-c",
                "app development", "mobile ui", "mobile ux", "push notifications", "responsive design", "pwa",
                "app store", "play store", "mobile testing", "cordova", "ionic", "capacitor", "adaptive layout"
            ],
            "design": [
                "design", "ui", "ux", "user experience", "user interface", "wireframe", "prototype", "figma", "sketch",
                "adobe", "photoshop", "illustrator", "indesign", "accessibility", "typography", "color theory",
                "layout", "responsive design", "user research", "usability testing", "information architecture"
            ],
            "management": [
                "management", "leadership", "team", "agile", "scrum", "kanban", "project", "product", "roadmap",
                "okr", "kpi", "sprint", "backlog", "milestone", "delivery", "stakeholder", "requirement",
                "specification", "meeting", "one-on-one", "retrospective", "planning", "delegation", "mentoring"
            ],
            "security": [
                "security", "cybersecurity", "encryption", "firewall", "penetration testing", "vulnerability",
                "hacking", "csrf", "xss", "sql injection", "authentication", "authorization", "oauth", "jwt",
                "virus", "malware", "ransomware", "phishing", "social engineering", "threat", "risk", "compliance"
            ],
            "soft_skills": [
                "communication", "teamwork", "problem solving", "critical thinking", "analytical", "creativity",
                "time management", "organization", "adaptability", "flexibility", "resilience", "empathy",
                "negotiation", "presentation", "public speaking", "writing", "reporting", "self-motivation",
                "patience", "conflict resolution", "stress management", "emotional intelligence", "feedback"
            ],
            "business": [
                "business", "strategy", "marketing", "finance", "sales", "operations", "consulting",
                "seo", "sem", "social media", "content", "campaign", "advertising", "conversion", "funnel",
                "roi", "kpi", "market analysis", "customer", "client", "account", "revenue", "profit", "budget",
                "forecast", "analysis", "report", "proposal", "pitch", "negotiation", "contract", "pricing"
            ],
            "education": [
                "teaching", "learning", "education", "curriculum", "instruction", "pedagogy", "student",
                "school", "university", "college", "degree", "certificate", "course", "workshop", "training",
                "mentor", "tutor", "academic", "research", "thesis", "dissertation", "publication", "journal"
            ],
            "healthcare": [
                "healthcare", "medical", "clinical", "patient", "care", "treatment", "diagnosis", "health",
                "hospital", "doctor", "nurse", "therapy", "pharmaceutical", "medicine", "drug", "vaccine",
                "surgery", "rehabilitation", "wellness", "preventive", "telemedicine", "electronic health record"
            ]
        }
        
        # Combine all category keywords into a main vocabulary
        self.vocabulary = []
        for category, keywords in self.skill_categories.items():
            self.vocabulary.extend(keywords)
        
        # Remove duplicates while preserving order
        self.vocabulary = list(dict.fromkeys(self.vocabulary))
        
        # Define linguistic feature names for NLP
        self.nlp_features = [
            "text_length",          # Total text length
            "word_count",           # Number of words
            "unique_word_ratio",    # Lexical diversity
            "avg_word_length",      # Average word length
            "avg_sentence_length",  # Average sentence length
            "education_level",      # Education keywords presence
            "experience_years",     # Experience span mentioned
            "technical_density",    # Density of technical terms
            "soft_skills_density",  # Density of soft skills terms
            "keyword_diversity"     # Variety of domains covered
        ]
        
        # Semantic concepts for career domains
        self.semantic_domains = list(self.skill_categories.keys())
        
        # Calculate embedding dimension: vocabulary + NLP features + semantic domains
        self.embedding_dim = len(self.vocabulary) + len(self.nlp_features) + len(self.semantic_domains)
        logger.info(f"Enhanced CareerRecommender initialized with embedding dimension: {self.embedding_dim}")
    
    def get_embedding(self, text):
        """
        Convert resume text to embedding vector using enhanced NLP approach
        
        Args:
            text (str): Resume or skills description text
            
        Returns:
            torch.Tensor: Enhanced embedding vector of the text with NLP features
        """
        logger.debug(f"Generating embedding for text of length: {len(text)}")
        
        try:
            # Clean up text (remove extra whitespace, normalize, lowercase)
            text = ' '.join(text.split()).lower()
            
            # 1. KEYWORD FEATURES: Generate bag-of-words embedding for technical keywords
            keyword_embedding = np.zeros(len(self.vocabulary), dtype=np.float32)
            
            # Count occurrences of each keyword in the text
            keyword_counts = {}
            for i, keyword in enumerate(self.vocabulary):
                # Count occurrences with word boundary check
                count = len(re.findall(r'\b' + re.escape(keyword) + r'[a-z]*\b', text))
                if count > 0:
                    keyword_embedding[i] = count
                    keyword_counts[keyword] = count
            
            # 2. NLP FEATURES: Extract linguistic patterns from the text
            words = re.findall(r'\b\w+\b', text)
            word_count = len(words)
            unique_words = set(words)
            unique_word_count = len(unique_words)
            
            # Text structure analysis
            sentences = re.split(r'[.!?]+', text)
            sentence_count = max(1, len([s for s in sentences if s.strip()]))
            
            # Average lengths
            avg_word_length = sum(len(word) for word in words) / max(1, word_count)
            avg_sentence_length = word_count / sentence_count
            
            # Lexical diversity (unique words / total words)
            lexical_diversity = unique_word_count / max(1, word_count)
            
            # Education level indicators
            education_terms = {
                "high school": 0.2, "associate": 0.4, "bachelor": 0.6, "master": 0.8, 
                "phd": 1.0, "doctorate": 1.0, "mba": 0.8, "degree": 0.5
            }
            education_level = max([education_terms.get(term, 0) 
                                  for term in education_terms.keys() 
                                  if term in text] or [0])
            
            # Experience indicators (years mentioned)
            experience_years = [int(match) for match in re.findall(r'\b(\d+)[\s-]*years?\b', text)]
            experience_score = min(1.0, max(experience_years) / 20 if experience_years else 0)
            
            # Technical vs soft skills density
            technical_keywords = set()
            for cat in ["programming", "data_science", "web_development", "devops", "databases"]:
                if cat in self.skill_categories:
                    technical_keywords.update(self.skill_categories[cat])
            
            soft_keywords = set(self.skill_categories.get("soft_skills", []))
            
            tech_count = sum(1 for word in unique_words if word in technical_keywords)
            soft_count = sum(1 for word in unique_words if word in soft_keywords)
            
            tech_density = min(1.0, tech_count / max(50, unique_word_count))
            soft_density = min(1.0, soft_count / max(20, unique_word_count))
            
            # Domain diversity (how many different skill domains are mentioned)
            domain_hits = 0
            for category in self.skill_categories:
                if any(kw in text for kw in self.skill_categories[category]):
                    domain_hits += 1
            domain_diversity = domain_hits / len(self.skill_categories)
            
            # Create NLP features vector
            nlp_embedding = np.array([
                min(1.0, len(text) / 5000),       # text_length
                min(1.0, word_count / 1000),      # word_count
                lexical_diversity,                # unique_word_ratio
                min(1.0, avg_word_length / 10),   # avg_word_length
                min(1.0, avg_sentence_length / 30), # avg_sentence_length
                education_level,                  # education_level
                experience_score,                 # experience_years
                tech_density,                     # technical_density
                soft_density,                     # soft_skills_density
                domain_diversity                  # keyword_diversity
            ], dtype=np.float32)
            
            # 3. SEMANTIC DOMAIN FEATURES: Calculate domain probabilities
            domain_embedding = np.zeros(len(self.semantic_domains), dtype=np.float32)
            
            for i, domain in enumerate(self.semantic_domains):
                if domain in self.skill_categories:
                    domain_keywords = self.skill_categories[domain]
                    # Count domain keyword matches
                    matches = sum(text.count(kw) for kw in domain_keywords)
                    # Calculate domain probability
                    domain_prob = min(1.0, matches / max(10, len(domain_keywords)))
                    domain_embedding[i] = domain_prob
            
            # 4. COMBINE ALL FEATURES
            combined_embedding = np.concatenate([
                keyword_embedding, 
                nlp_embedding, 
                domain_embedding
            ])
            
            # Normalize the embedding
            norm = np.linalg.norm(combined_embedding)
            if norm > 0:
                combined_embedding = combined_embedding / norm
                
            # Convert to PyTorch tensor
            embedding_tensor = torch.tensor(combined_embedding, dtype=torch.float32)
            logger.debug(f"Generated enhanced embedding of shape: {embedding_tensor.shape}")
            
            return embedding_tensor
        except Exception as e:
            logger.error(f"Error generating enhanced embedding: {str(e)}")
            raise
