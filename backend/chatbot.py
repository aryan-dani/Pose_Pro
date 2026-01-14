"""
Chatbot module for PosePro
Simple rule-based chatbot for shoulder raise form questions
"""

import datetime


# Chatbot response rules
CHATBOT_RESPONSES = {
    'greeting': {
        'keywords': ['hello', 'hi', 'hey'],
        'response': "Hello! I'm your shoulder raise assistant. How can I help you with your lateral raise form?"
    },
    'form': {
        'keywords': ['form', 'technique', 'proper'],
        'response': "Proper lateral raise form: Stand tall, keep elbows slightly bent, raise arms to shoulder height, avoid shrugging shoulders, control the movement both up and down."
    },
    'mistakes': {
        'keywords': ['common mistake', 'error', 'wrong'],
        'response': "Common mistakes: 1) Using too much weight 2) Shrugging shoulders 3) Swinging the body 4) Raising arms too high 5) Bending elbows excessively."
    },
    'elbow': {
        'keywords': ['elbow', 'bend'],
        'response': "Keep a slight bend in elbows (160-180°). Don't lock them out completely or bend too much. This protects your joints."
    },
    'height': {
        'keywords': ['shoulder height', 'how high'],
        'response': "Raise arms to shoulder height (parallel to ground). Going higher can impinge shoulder joint."
    },
    'scoring': {
        'keywords': ['score', 'grading', 'evaluation'],
        'response': "Your form is evaluated on: ROM (70-90° ideal), symmetry (<5° difference), torso stability, movement smoothness, and elbow position."
    },
    'pain': {
        'keywords': ['pain', 'hurt', 'injury'],
        'response': "If you feel pain, stop immediately. Reduce weight, check form, and consider consulting a professional. Never work through joint pain."
    },
    'benefits': {
        'keywords': ['benefit', 'why', 'purpose'],
        'response': "Lateral raises target medial deltoids, improving shoulder width and strength. Helps with shoulder stability and posture."
    },
    'weight': {
        'keywords': ['weight', 'heavy', 'light'],
        'response': "Start light to master form. Weight should allow controlled movement without swinging. Quality over quantity!"
    },
    'reps': {
        'keywords': ['rep', 'count', 'how many'],
        'response': "Aim for 8-15 reps per set with good form. 3-4 sets total. Rest 60-90 seconds between sets."
    },
    'thanks': {
        'keywords': ['thank', 'thanks'],
        'response': "You're welcome! Keep up the good work. Remember, consistency with proper form is key!"
    }
}

DEFAULT_RESPONSE = "I'm here to help with shoulder raise questions! Ask me about form, technique, common mistakes, or your performance scores."


def get_chatbot_response(message):
    """
    Generate a response to user message based on keyword matching.
    
    Args:
        message: User's message string
    
    Returns:
        str: Bot's response
    """
    message_lower = message.lower()
    
    # Check each response category
    for category, data in CHATBOT_RESPONSES.items():
        if any(keyword in message_lower for keyword in data['keywords']):
            return data['response']
    
    return DEFAULT_RESPONSE


class ChatHistory:
    """Manages in-memory chat history."""
    
    def __init__(self):
        self.messages = []
    
    def add_message(self, role, message):
        """
        Add a message to chat history.
        
        Args:
            role: 'user', 'bot', or 'system'
            message: Message content
        """
        self.messages.append({
            'role': role,
            'message': message,
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S')
        })
    
    def get_history(self):
        """Return all chat messages."""
        return self.messages
    
    def clear(self):
        """Clear chat history."""
        self.messages = []


# Global chat history instance
chat_history = ChatHistory()
