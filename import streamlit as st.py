import streamlit as st
import pandas as pd
import re
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Page Config
st.set_page_config(page_title="AI News Detector", page_icon="🔍")

# 2. Text Cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W"," ",text) 
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)    
    return text

# 3. Training Logic (Balanced)
@st.cache_resource
def load_and_train():
    f_path = "Fakedata.csv" if os.path.exists("Fakedata.csv") else "Fakedata.csv.csv"
    t_path = "Truedata.csv" if os.path.exists("Truedata.csv") else "Truedata.csv.csv"
    
    fake = pd.read_csv(f_path)
    real = pd.read_csv(t_path)
    
    # We balance the data so it doesn't always guess "Fake"
    fake = fake.sample(n=min(len(fake), len(real)), random_state=42)
    
    fake["label"], real["label"] = 0, 1
    data = pd.concat([fake, real]).reset_index(drop=True)
    data['text'] = data['text'].apply(clean_text)
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(data["text"])
    
    # Logistic Regression with balanced weights
    model = LogisticRegression(class_weight='balanced')
    model.fit(X, data["label"])
    return vectorizer, model

# 4. App UI
st.title("🔍 AI Fake News Detector")

# Initialize Session State for the Refresh button
if "my_text" not in st.session_state:
    st.session_state.my_text = ""

vectorizer, model = load_and_train()

# The Input Box
news_text = st.text_area("Paste News Content Here:", value=st.session_state.my_text, height=200)

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("Analyze News"):
        if news_text.strip():
            cleaned = clean_text(news_text)
            vec = vectorizer.transform([cleaned])
            prediction = model.predict(vec)[0]
            
            if prediction == 1:
                st.success("### Result: REAL ✅")
            else:
                st.error("### Result: FAKE ❌")
        else:
            st.warning("Please enter some text!")

with col2:
    if st.button("🔄 Refresh / Clear"):
        # This wipes the input clean
        st.session_state.my_text = "" 
        st.rerun()

st.markdown("---")