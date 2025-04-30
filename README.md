from flask import Flask, request, jsonify
import os
import nltk
import ssl
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL and NLTK setup
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Define intents
intents = [ ... ]  # Keep your intents list unchanged

# Model training
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot_response(input_text):
    input_vec = vectorizer.transform([input_text])
    tag = clf.predict(input_vec)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Flask App
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    response = chatbot_response(user_message)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

Save the code in a file, e.g., flask_chatbot.py.

Run the API: python flask_chatbot.py

Test with a POST request (e.g., via Postman or curl): curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"message": "Hi"}'


