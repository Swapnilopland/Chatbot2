from flask import Flask, request, jsonify
import joblib
import spacy
import re
from dateparser import parse as parse_date
from dateparser.search import search_dates
from datetime import datetime, timedelta
import random

# Load trained models
main_classifier = joblib.load("main_intent_classifier.pkl")
greeting_classifier = joblib.load("greeting_sub_classifier.pkl")
#intent_mapping = joblib.load("intent_mapping.pkl")
#reverse_intent_mapping = {v: k for k, v in intent_mapping.items()}
leave_type_model = joblib.load("leave_type_predictor.pkl")
leave_vectorizer = joblib.load("leave_vectorizer.pkl")
faq_c_model = joblib.load("faq_model.pkl")

# Load NER model
ner_model = spacy.load("leave_entity_recognition_model")

# Leave type mapping
LEAVE_TYPES_MAP = {
    "casual": "Casual Leave",
    "cl": "Casual Leave",
    "sick": "Sick Leave",
    "sl": "Sick Leave",
    "earned": "Earned Leave",
    "el": "Earned Leave",
    "maternity": "Maternity Leave",
    "ml": "Maternity Leave",
    "paternity": "Paternity Leave",
    "pl": "Paternity Leave"
}

# Greeting responses
greeting_responses = {
    "greeting_morning": ["Good morning! 🌞", "Morning vibes! 😊"],
    "greeting_afternoon": ["Good afternoon! ☀️", "Hello! Hope your day is going well."],
    "greeting_evening": ["Good evening! 🌇", "Relax and enjoy your evening 😊"],
    "greeting_general": ["Hello there! 👋", "Hey! How can I assist you today?"]
}

# Function to extract leave type from message
def extract_leave_type(text):
    text_lower = text.lower()
    for keyword, full_form in LEAVE_TYPES_MAP.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            return full_form
    return None

# Function to predict leave type if not explicitly mentioned
def predict_leave_type(text):
    if not text:
        return None
    try:
        text_tfidf = leave_vectorizer.transform([text])
        predicted_type = leave_type_model.predict(text_tfidf)[0]
        return LEAVE_TYPES_MAP.get(predicted_type.lower(), predicted_type)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Function to extract reason from text
def extract_reason(text):
    pattern = r'(?:because(?: of)?|coz|cuz|cause|due to|as|since|for|so|that(?: is|’s)? why|in order to|to|on account of)\s+(.*?)(?:[.,;]|$)'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else None

# Function to extract dates from text
def extract_dates(text):
    from_date, to_date = None, None
    range_match = re.search(r'(\d{1,2})\s*(?:to|-)\s*(\d{1,2})\s*([A-Za-z]+)?\s*(\d{4})?', text)
    
    if range_match:
        day1, day2, month, year = range_match.groups()
        if not month or not year:
            date_matches = search_dates(text)
            if date_matches:
                ref_date = date_matches[0][1]
                month = month or ref_date.strftime("%B")
                year = year or ref_date.year
        from_date = parse_date(f"{day1} {month} {year}").strftime("%Y-%m-%d")
        to_date = parse_date(f"{day2} {month} {year}").strftime("%Y-%m-%d")
    
    else:
        date_matches = search_dates(text)
        if date_matches:
            unique_dates = list(set(match[1] for match in date_matches))
            if len(unique_dates) == 1:
                from_date = to_date = unique_dates[0].strftime("%Y-%m-%d")
            elif len(unique_dates) >= 2:
                from_date, to_date = unique_dates[0].strftime("%Y-%m-%d"), unique_dates[1].strftime("%Y-%m-%d")

    return from_date, to_date

# Function to extract entities using NER model
def extract_entities(text, intent=None):
    entities = {}

    if not intent:
        X_text = leave_vectorizer.transform([text]).toarray()
        intent = main_classifier.predict(X_text)[0]

    if intent == "apply_leave":
        doc = ner_model(text)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                entities["date"] = ent.text
            elif ent.label_ == "LEAVE_TYPE":
                entities["leave_type"] = ent.text
            elif ent.label_ == "REASON":
                entities["reason"] = ent.text

        if "tomorrow" in text.lower():
            entities["date"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "yesterday" in text.lower():
            entities["date"] = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        if "date" in entities:
            entities["begin_date"], entities["end_date"] = extract_dates(entities["date"])

    return entities

# Function to determine greeting response based on time
def get_greeting_response(sub_intent=None):
    current_hour = datetime.now().hour
    if not sub_intent:
        if current_hour < 12:
            sub_intent = "greeting_morning"
        elif 12 <= current_hour < 18:
            sub_intent = "greeting_afternoon"
        else:
            sub_intent = "greeting_evening"
    return random.choice(greeting_responses.get(sub_intent, greeting_responses["greeting_general"]))

# Process user message and classify intent
""" def process_message(message):
    intent = main_classifier.predict([message])[0]

    if intent == "greetings":
        sub_intent = greeting_classifier.predict([message])[0]
        return {"intent": sub_intent, "entities": {}, "response": get_greeting_response(sub_intent)}
    
    elif intent == "faq":
        faq_response = faq_c_model.predict([message])[0]
        return {"intent": "faq", "entities": {}, "response": faq_response}

    elif intent == "apply_leave":
        entities = extract_entities(message, intent)
        
        if "leave_type" not in entities or not entities["leave_type"]:
            extracted_leave_type = extract_leave_type(message)
            if not extracted_leave_type:
                extracted_leave_type = predict_leave_type(message)
            entities["leave_type"] = extracted_leave_type

        if "purpose" not in entities or not entities["purpose"]:
            entities["purpose"] = extract_reason(message)

        if "begin_date" not in entities or not entities["begin_date"]:
            entities["begin_date"], entities["end_date"] = extract_dates(message)

        return {
            "intent": intent,
            "entities": entities
        }

    return {"intent": "unknown", "entities": {}, "response": "Sorry, I didn’t quite understand that."} """

def process_message(message):
    intent = main_classifier.predict([message])[0]
    print(f"🛠 DEBUG: Message: '{message}', Predicted Intent: '{intent}'")  # Debugging Log

    if intent == "greetings":
        sub_intent = greeting_classifier.predict([message])[0]
        return {"intent": sub_intent, "entities": {}, "response": get_greeting_response(sub_intent)}
    
    elif intent == "faq":
        faq_response = faq_c_model.predict([message])[0]
        print(f"🛠 DEBUG: FAQ Response: {faq_response}")  # Debugging Log
        return {"intent": "faq", "entities": {}, "response": faq_response}

    elif intent == "apply_leave":
        entities = extract_entities(message, intent)
        
        if "leave_type" not in entities or not entities["leave_type"]:
            extracted_leave_type = extract_leave_type(message)
            if not extracted_leave_type:
                extracted_leave_type = predict_leave_type(message)
            entities["leave_type"] = extracted_leave_type

        if "purpose" not in entities or not entities["purpose"]:
            entities["purpose"] = extract_reason(message)

        if "begin_date" not in entities or not entities["begin_date"]:
            entities["begin_date"], entities["end_date"] = extract_dates(message)

        return {
            "intent": intent,
            "entities": entities
        }
    
    elif intent == "cancel_leave":
        from_date, to_date = extract_dates(message)

        return {
            "intent": "cancel_leave",
            "entities": {
                "begin_date": from_date,
                "end_date": to_date,
            },
            "response": f"Leave cancellation request noted for dates: {from_date} to {to_date}."
                    }
    
    elif intent == "fetch_balance":
        
        return {
            "intent": "fetch_balnce",
            "response": f"leaves available"
                    }
    
    else:
        return {"intent": "unknown", "entities": {}, "response": "Sorry, I didn’t quite understand that."}

    #return {"intent": "unknown", "entities": {}, "response": "Sorry, I didn’t quite understand that."}


""" # API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({"error": "Message is required"}), 400

    result = process_message(message)
    return jsonify(result)

# Run Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) """
  

test_messages = [
    "I want to apply leave on 25th March.",
    "Apply leave for tomorrow.",
    "Apply one day leave on 5th April.",
    "I need a day off on 28th March due to a doctor's appointment.",
    "I’d like to apply for a sick leave for today.",
    "Can you apply for casual leave for me from May 3rd to May 5th? Reason: personal work.",
    "Apply for half-day leave tomorrow morning as I have a doctor’s appointment.",
    "I need a day off this Friday for a family function.",
    "Please apply for earned leave from 12th to 14th April due to travel plans.",
    "Can you apply for a leave for yesterday? I wasn’t feeling well.",
    "I want to take a leave on April 25 for a wedding.",
    "I have some urgent work so I am out of station so I need one week leave.", 
    "Need two days off 19, 20 Aprils",
    "I am going home so 7 days leave apply",
    "Sick Leave apply for three day",
    "I want to apply leave on 25th March.",
    "Apply leave for tomorrow.",
    "Apply one day leave on 5th April.",
    "I need a day off on 28th March due to a doctor's appointment.",
    "Please apply leave for 2nd April – I have a family function.",
    "Apply leave on Friday, I'm not feeling well.",
    "Apply casual leave on 30th March.",
    "I want to take sick leave tomorrow.",
    "Please apply earned leave from 3rd to 5th April.",
    "Steps to apply for sick leave?",
    "How can I reset my password?",
    "What is my remaining leave balance?",
    "Apply 2 days leave from 12th to 13th April due to a wedding.",
    "Please apply leave from 5th to 7th April, I’m traveling.",
    "Need 3 days sick leave from 10th to 12th March.",
    "Apply leave this Thursday.",
    "I’ll be on leave next Monday and Tuesday.",
    "Apply leave for this weekend.",
    "Apply leave for 1st May in advance – I have a personal engagement.",
    "Schedule my leave on 15th August.",
    "Hi there!",
    "Hello!",
    "Hey!",
    "Good morning!",
    "Good evening!" 
]
  
#Loop through test messages
for msg in test_messages:
    print(f"Input: {msg}")
    output = process_message(msg)
    print(f"Output: {output}\n") 

#print(process_message(message)) 





