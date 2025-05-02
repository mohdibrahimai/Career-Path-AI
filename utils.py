import logging

logger = logging.getLogger(__name__)

def get_career_list():
    """
    Returns a list of possible career paths
    
    Returns:
        list: List of career titles
    """
    # This list can be expanded with more career options
    return [
        "Data Scientist",
        "Software Engineer",
        "AI Engineer",
        "Machine Learning Engineer",
        "Product Manager",
        "UX Designer",
        "DevOps Engineer",
        "Cloud Architect",
        "Cybersecurity Analyst",
        "Business Analyst",
        "Web Developer",
        "Mobile App Developer",
        "Database Administrator",
        "Network Engineer",
        "Project Manager",
        "Digital Marketing Specialist",
        "Systems Administrator",
        "Data Engineer",
        "Full Stack Developer",
        "QA Engineer",
        "Technical Writer",
        "Blockchain Developer",
        "AR/VR Developer",
        "Game Developer",
        "IoT Engineer"
    ]

def get_career_info(career_title):
    """
    Returns detailed information about a specific career
    
    Args:
        career_title (str): Title of the career
        
    Returns:
        dict: Career information including description, skills, and salary
    """
    career_info = {
        "Data Scientist": {
            "description": "Analyzes and interprets complex data to help organizations make better decisions.",
            "skills": ["Python", "R", "SQL", "Machine Learning", "Statistics", "Data Visualization"],
            "salary_range": "$90,000 - $150,000",
            "growth_outlook": "High demand with 36% growth through 2031"
        },
        "Software Engineer": {
            "description": "Designs, develops, and maintains software systems and applications.",
            "skills": ["Programming", "Algorithms", "Data Structures", "Testing", "Version Control"],
            "salary_range": "$80,000 - $140,000",
            "growth_outlook": "Steady growth with 25% increase through 2031"
        },
        "AI Engineer": {
            "description": "Builds AI systems and models that can perform tasks requiring human intelligence.",
            "skills": ["Machine Learning", "Deep Learning", "NLP", "Python", "TensorFlow/PyTorch"],
            "salary_range": "$100,000 - $160,000",
            "growth_outlook": "Rapid growth with 40% increase through 2031"
        },
        "Machine Learning Engineer": {
            "description": "Creates and implements machine learning models to solve business problems.",
            "skills": ["Python", "Machine Learning", "Deep Learning", "Data Modeling", "Distributed Computing"],
            "salary_range": "$95,000 - $155,000",
            "growth_outlook": "Very high demand with 45% growth through 2031"
        },
        "Product Manager": {
            "description": "Guides the success of a product throughout its lifecycle, from strategy to launch.",
            "skills": ["Product Strategy", "User Research", "Market Analysis", "Roadmapping", "Communication"],
            "salary_range": "$85,000 - $145,000",
            "growth_outlook": "Strong growth with 20% increase through 2031"
        },
        "UX Designer": {
            "description": "Creates meaningful and relevant experiences for users through design thinking.",
            "skills": ["User Research", "Wireframing", "Prototyping", "Visual Design", "Usability Testing"],
            "salary_range": "$75,000 - $130,000",
            "growth_outlook": "Growing field with 23% increase through 2031"
        },
        "DevOps Engineer": {
            "description": "Combines development and operations to improve and streamline software delivery.",
            "skills": ["CI/CD", "Cloud Platforms", "Infrastructure as Code", "Containerization", "Monitoring"],
            "salary_range": "$90,000 - $140,000",
            "growth_outlook": "High demand with 30% growth through 2031"
        },
        "Cloud Architect": {
            "description": "Designs and oversees cloud computing strategies and implementations.",
            "skills": ["AWS/Azure/GCP", "Cloud Security", "Networking", "Distributed Systems", "Containerization"],
            "salary_range": "$110,000 - $170,000",
            "growth_outlook": "Strong growth with 25% increase through 2031"
        },
        "Cybersecurity Analyst": {
            "description": "Protects systems, networks, and data from cyber threats and attacks.",
            "skills": ["Security Tools", "Threat Detection", "Risk Assessment", "Incident Response", "Security Protocols"],
            "salary_range": "$85,000 - $140,000",
            "growth_outlook": "Critical demand with 35% growth through 2031"
        },
        "Business Analyst": {
            "description": "Analyzes business processes and systems to recommend improvements and solutions.",
            "skills": ["Requirements Gathering", "Process Modeling", "Data Analysis", "Documentation", "Communication"],
            "salary_range": "$70,000 - $120,000",
            "growth_outlook": "Steady growth with 15% increase through 2031"
        },
        "Web Developer": {
            "description": "Creates and maintains websites and web applications.",
            "skills": ["HTML/CSS", "JavaScript", "Responsive Design", "Web Frameworks", "API Integration"],
            "salary_range": "$65,000 - $120,000",
            "growth_outlook": "Consistent demand with 23% growth through 2031"
        },
        "Mobile App Developer": {
            "description": "Designs and builds applications for mobile devices.",
            "skills": ["iOS/Android Development", "Swift/Kotlin/React Native", "UI Design", "API Integration", "App Store Deployment"],
            "salary_range": "$75,000 - $135,000",
            "growth_outlook": "Strong demand with 25% growth through 2031"
        },
    }
    
    # Default info for careers not explicitly defined
    default_info = {
        "description": f"A professional who specializes in {career_title.lower()} tasks and responsibilities.",
        "skills": ["Technical Knowledge", "Problem Solving", "Communication", "Teamwork", "Continuous Learning"],
        "salary_range": "$70,000 - $130,000",
        "growth_outlook": "Positive growth projected through 2031"
    }
    
    return career_info.get(career_title, default_info)
