from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random

app = Flask(__name__)

# Sample intents
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there!", "Hello!", "Hey!", "I'm fine, thank you!", "Nothing much!"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye!", "See you later!", "Take care!"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome!", "No problem!", "Glad I could help!"]
    },
    {
        "tag": "faq",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "leave",
        "patterns": ["I want to apply for leave", "Can I take sick leave", "Need a vacation", "Apply for casual leave"],
        "responses": ["Leave request noted.", "You can apply for leave using the HR portal.", "Leave applied successfully."]
    }
]

# Training model
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
clf.fit(x, tags)

def predict_tag(text):
    vec = vectorizer.transform([text])
    return clf.predict(vec)[0]

def get_response(tag):
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

def extract_entities(text):
    text = text.lower()
    entities = {}
    if "tomorrow" in text:
        entities["date"] = "tomorrow"
    if "monday" in text:
        entities["day"] = "Monday"
    if "fever" in text:
        entities["reason"] = "fever"
    if "vacation" in text:
        entities["reason"] = "vacation"
    return entities

def detect_subintent(text):
    text = text.lower()
    if "vacation" in text:
        return "vacation request"
    elif "sick" in text:
        return "sick request"
    else:
        return "general request"

@app.route('/')
def index():
    return "Chatbot API is running."

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Text input required"}), 400

    intent = predict_tag(text)

    result = {"intent": intent}

    # Routing based on intent
    if intent == "leave":
        result["subintent"] = detect_subintent(text)
        result["entities"] = extract_entities(text)
        result["response"] = get_response(intent)
    elif intent in ["faq", "greeting", "goodbye", "thanks", "help"]:
        result["response"] = get_response(intent)
    else:
        result["response"] = "Sorry, I couldn't understand your request."

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
