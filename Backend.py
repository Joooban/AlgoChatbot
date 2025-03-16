# Import required libraries
import pandas as pd
import google.generativeai as genai
import os
import nltk
import streamlit as st
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Download necessary NLTK data
nltk.download('words')

# Load environment variables
load_dotenv()
api_key = os.getenv('API_KEY')
genai.configure(api_key=api_key)

# Greeting and Goodbye Keywords
GREETING_KEYWORDS = ["hi", "hello", "hey", "greetings", "what's up", "yo", "how are you"]
GOODBYE_KEYWORDS = ["thank you", "goodbye", "thanks", "bye"]

def load_dataset(file_path):
    df = pd.read_csv(file_path, usecols=["text", "label"], low_memory=False)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df["text"] = df["text"].fillna("")

    X = df["text"]
    y = df["label"]

    model = make_pipeline(TfidfVectorizer(stop_words="english", max_features=3000), MultinomialNB())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model.fit(X_train, y_train)

    # Compute evaluation metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return df, model, accuracy, precision, recall, f1


# Classify user input
def classify_news(model, text):
    prediction = model.predict([text])[0]
    # Get probability scores for each class
    probabilities = model.predict_proba([text])[0]
    fake_prob = probabilities[0]
    real_prob = probabilities[1]
    
    result = "Fake News" if prediction == 0 else "Real News"
    
    return result, fake_prob, real_prob

# Function to search dataset based on user query
def search_dataset(df, query):
    query = query.lower()
    matching_rows = df[df.apply(lambda row: row.astype(str).str.contains(query, case=False, na=False).any(), axis=1)]

    if not matching_rows.empty:
        return matching_rows.to_string(index=False)
    else:
        return "Sorry, I couldn't find relevant information in the dataset."

# Function to query Gemini AI
def query_gemini_api(model, df, user_input):
    # Simple conversational responses
    if any(word in user_input.lower() for word in GREETING_KEYWORDS):
        return "Hello! How can I assist you today?", None
    if any(word in user_input.lower() for word in GOODBYE_KEYWORDS):
        return "You're welcome! Have a great day!", None
    
    # Classify longer inputs as news articles
    if len(user_input.split()) > 20:  # More realistic threshold
        classification = classify_news(model, user_input)
        return f"The news article is classified as: **{classification}**", classification
        
    # For shorter queries, use Gemini with dataset context
    dataset_info = search_dataset(df, user_input)
    if dataset_info == "Sorry, I couldn't find relevant information in the dataset.":
        # No relevant data found, just use Gemini directly
        context = "Provide information about fake news detection"
    else:
        context = f"Based on this dataset information: {dataset_info}"
    
    ai_model = genai.GenerativeModel("models/gemini-1.5-flash-8b")
    response = ai_model.generate_content([
        f"Provide a concise and accurate response. {context}, answer the query: {user_input}."
    ])
    
    return response.text if response else "I couldn't generate a response.", None

# Function to handle Streamlit Chat UI
def handle_conversation(dataset_path):
    df, model, accuracy, precision, recall, f1 = load_dataset(dataset_path)

    if "classification_history" not in st.session_state:
        st.session_state.classification_history = []

    user_input = st.chat_input("Paste a news article to classify...")

    if user_input:
        # Classify the article and get probabilities
        classification, fake_prob, real_prob = classify_news(model, user_input)
        
        # Store in history with timestamp and probabilities
        import datetime
        st.session_state.classification_history.append({
            "text": user_input,
            "classification": classification,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Clear previous display
        st.empty()
        
        # Display all classifications sequentially
        for item in st.session_state.classification_history:
            # Display user inputted article
            st.subheader("Article")
            st.markdown(f"{item['text']}")
            
            # Display classification result with probabilities
            st.subheader("Classification Result")
            st.write(f"**{item['classification']}**")
            
            # Display prediction probabilities
            st.subheader("Prediction Confidence")
            st.write(f"Probability of Fake News: **{item['fake_probability']*100:.2f}%**")
            st.write(f"Probability of Real News: **{item['real_probability']*100:.2f}%**")
            
            # Add a separator between entries
            st.markdown("---")
        
        # Display overall model performance metrics only once at the bottom
        st.header("Overall Model Performance")
        st.write(f"✅ **Accuracy:** {accuracy * 100:.2f}%")
        st.write(f"✅ **Precision:** {precision * 100:.2f}%")
        st.write(f"✅ **Recall:** {recall * 100:.2f}%")
        st.write(f"✅ **F1-score:** {f1 * 100:.2f}%")
