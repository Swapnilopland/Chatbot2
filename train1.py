import pandas as pd
import joblib
import random
import spacy
from spacy.training import Example
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# ======== Step 1: Train Main Intent Classifier ========

data = {
    "text": [
        # Apply Leave (31)
        "I want to apply for leave", "Can I take a day off?", "leave", "Need sick leave for tomorrow",
        "Apply leave for me", "Please apply leave", "Apply sick leave", "I need to take leave",
        "Can you apply leave for me on 23rd Feb?", "I was sick, need leave",
        "Apply leave as I was not feeling well", "Apply leave on 23rd Feb due to sickness",
        "Take a leave on my behalf", "I need a day off on 23rd Feb",
        "I want to request a sick leave", "I want to apply leave because I was ill",
        "Can I take a medical leave?", "I was not well yesterday",
        "Mark me on leave for yesterday", "Sick leave for yesterday please",
        "Feeling unwell, apply leave", "I had fever, apply leave", "Please record a sick leave",
        "Apply one-day leave", "Leave request for 23 Feb", "Apply leave due to health issue",
        "I wasn’t feeling well yesterday", "Can you apply one day leave?", "Need leave for 23rd Feb",
        "Leave required for 23 Feb due to sickness", "Apply my leave for yesterday",
        
        # Greetings (11)
        "Hello!", "Hi there", "Hey", "Hey, how are you?", "Good morning",
        "What's up?", "Hello buddy!", "Greetings", "Hey there!", "Hi", "good evening",
        
        # Balance (10)
        "What is my balance?", "Show me my account balance",
        "How much money do I have?", "Can you tell me my balance?",
        "What's my available leave?", "Tell me my remaining balance",
        "How many leaves are left?", "Leave balance please",
        "What’s my leave status?", "How many sick leaves do I have left?",
        
        # FAQ (16)
        "How do I apply for leave?", "What is the process to request a leave?",
        "Steps to apply for sick leave?", "How can I reset my password?",
        "What is my remaining leave balance?", "How do I change my profile details?",
        "Can you tell me how to apply for a day off?", "What is the process for applying maternity leave?",
        "How to check my work schedule?", "Where can I see my approved leaves?",
        "How can I update my email address?", "What is the HR policy on remote work?",
        "How to request a travel allowance?", "How to submit reimbursement claims?",
        "Where can I find my previous salary slips?", "How do I get IT support?",
        
        # Cancel Leave (23)
        "I want to cancel my leave", "Cancel my leave request", "Please cancel my applied leave",
        "I don’t need the leave anymore", "Revoke my leave application", "Withdraw the leave I submitted",
        "Cancel leave for next week", "Cancel my sick leave", "I won’t be needing the leave I requested",
        "Please withdraw my leave request", "Can you cancel the leave I applied for?", "I’d like to retract my leave application",
        "Cancel the leave request I made yesterday", "No longer need to take leave—please cancel it",
        "Please undo my leave request", "Cancel the leave scheduled for tomorrow", "Cancel leave from 25th to 27th",
        "Please cancel the casual leave I applied for", "I mistakenly applied for leave, cancel it",
        "Cancel my upcoming leave", "Please take back the leave request",
        "Can you remove my leave entry?", "I changed my mind—cancel the leave"
    ],
    "intent": (
        ["apply_leave"] * 31 +
        ["greetings"] * 11 +
        ["fetch_balance"] * 10 +
        ["faq"] * 16 +
        ["cancel_leave"] * 23 
    )
}


df = pd.DataFrame(data)

# Split data for main classifier
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["intent"], test_size=0.2, random_state=42,stratify=df["intent"])

# Train main intent classifier
main_classifier = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=5000))
main_classifier.fit(X_train, y_train)

# Save main classifier
joblib.dump(main_classifier, "main_intent_classifier.pkl")

# ======== Step 2: Train Greeting Sub-Classifier ========

greeting_data = {
    "text": ["Hi", "Hello there", "Hey", "Hello", "Good morning", "Morning!", "Good evening", "Evening!"],
    "sub_intent": ["greeting_general", "greeting_general", "greeting_general", "greeting_general",
                   "greeting_morning", "greeting_morning", "greeting_evening", "greeting_evening"]
}
greeting_df = pd.DataFrame(greeting_data)

greeting_classifier = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000)
)
greeting_classifier.fit(greeting_df['text'], greeting_df['sub_intent'])

# Save greeting classifier
joblib.dump(greeting_classifier, "greeting_sub_classifier.pkl")

# ======== Step 3: Train Leave Type Prediction Model ========

leave_data = [
    ("I have a fever and need rest.", "Sick Leave"),
    ("I am not feeling well today.", "Sick Leave"),
    ("Going on vacation next week.", "Casual Leave"),
    ("I need time off for a family function.", "Casual Leave"),
    ("My wife is expecting a baby soon.", "Maternity Leave"),
    ("Taking care of my newborn baby.", "Maternity Leave"),
    ("My wife just gave birth, need time off.", "Paternity Leave"),
    ("Need leave to take care of my father.", "Earned Leave"),
    ("I have an important personal engagement.", "Earned Leave"),
    ("Traveling out of town for a few days.", "Casual Leave"),
]

# Convert data to DataFrame
df_leave = pd.DataFrame(leave_data, columns=["Leave Request", "Leave Type"])

# Features (Text) and Labels (Leave Type)
X_leave = df_leave["Leave Request"]
y_leave = df_leave["Leave Type"]

# Convert text data into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf_leave = vectorizer.fit_transform(X_leave)

# Split data into training and test sets
X_train_leave, X_test_leave, y_train_leave, y_test_leave = train_test_split(X_tfidf_leave, y_leave, test_size=0.2, random_state=42)

# Train the model
leave_model = LogisticRegression()
leave_model.fit(X_train_leave, y_train_leave)

# Save the trained model and vectorizer
joblib.dump(leave_model, "leave_type_predictor.pkl")
joblib.dump(vectorizer, "leave_vectorizer.pkl")

print("Leave type prediction model saved successfully!")

# ======== Step 4: Train Named Entity Recognition (NER) Model ========

import spacy
from spacy.tokens import DocBin
from spacy.training import Example

# Load or create a blank NLP model
nlp = spacy.blank("en")

# Define training data (Ensure entity spans are correctly aligned)
TRAIN_DATA = [
    ("I have a fever and need rest.", {"entities": [(9, 14, "REASON"), (19, 23, "LEAVE_TYPE")]}),
    ("I am not feeling well today.", {"entities": [(5, 21, "REASON")]}),
    ("My wife is expecting a baby soon.", {"entities": [(8, 32, "REASON")]}),
    ("Traveling out of town for a few days.", {"entities": [(0, 9, "LEAVE_TYPE")]}),
    ("Taking care of my newborn baby.", {"entities": [(0, 11, "LEAVE_TYPE"), (15, 30, "REASON")]}),
]

# Function to validate and adjust entity offsets
def validate_offsets(nlp, text, entities):
    doc = nlp.make_doc(text)
    valid_spans = []
    
    for start, end, label in entities:
        span = doc.char_span(start, end, label=label)
        if span is None:
            print(f"⚠️ Misaligned entity: '{text[start:end]}' in '{text}'")
        else:
            valid_spans.append((span.start_char, span.end_char, label))
    
    return valid_spans

# Process training data and save it
db = DocBin()
for text, annotations in TRAIN_DATA:
    valid_entities = validate_offsets(nlp, text, annotations["entities"])
    doc = nlp.make_doc(text)
    ents = [doc.char_span(start, end, label) for start, end, label in valid_entities]
    doc.ents = [e for e in ents if e is not None]
    db.add(doc)

# Save preprocessed training data
db.to_disk("./train_data.spacy")

print("✅ NER training data processed and saved successfully!")

#

faq_data = {
    "question": [
        "How do I apply for leave?",
        "What is the process to request a leave?",
        "Steps to apply for sick leave?",
        "How can I reset my password?",
        "What is my remaining leave balance?",
        "How do I change my profile details?",
        "Can you tell me how to apply for a day off?",
        "What is the process for applying maternity leave?",
        "How to check my work schedule?"
    ],
    "answer": [
        "To apply for leave, navigate to the leave section, select 'Add New', fill in the details, and submit.",
        "You can request a leave by selecting 'Leave' from the menu, filling in the details, and clicking submit.",
        "To apply for sick leave, go to the leave section, choose 'Sick Leave', enter your reason, and submit.",
        "To reset your password, click 'Forgot Password' on the login page and follow the instructions.",
        "To check your leave balance, navigate to 'Leave' and select 'Leave Balance'.",
        "To update your profile, go to the settings section and edit your details.",
        "Click on the 'Leave' section, choose 'Add New', select the leave type, and submit your request.",
        "To apply for maternity leave, go to 'Leave', select 'Maternity Leave', and submit the required documents.",
        "You can check your work schedule by logging into the employee portal under 'Schedule'."
    ]
}

df_faq = pd.DataFrame(faq_data)


X_faq = df_faq["question"]
y_faq = df_faq["answer"]

faq_model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
faq_model.fit(X_faq, y_faq)
joblib.dump(faq_model, "faq_model.pkl")


