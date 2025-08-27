import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


model = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')


st.title("📩 SMS Spam Classifier")
st.markdown("Enter your SMS message below to check if it's spam.")


user_input = st.text_area("Type your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]

        if prediction == 1:
            st.error("🔴 This message is **SPAM**!")
        else:
            st.success("🟢 This message is **NOT SPAM**.")
