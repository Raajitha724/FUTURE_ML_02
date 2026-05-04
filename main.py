import pandas as pd
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Ticket Classifier", layout="wide")
st.title("🎫 Support Ticket Classification & Prioritization")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("customer_support_tickets.csv", encoding='latin1')
    df = df[['Ticket Description', 'Ticket Type', 'Ticket Priority']]
    df.dropna(inplace=True)
    return df

df = load_data()

# -------------------------------
# TEXT CLEANING
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = ' '.join(text.split())  # remove extra spaces
    return text

df['clean_text'] = df['Ticket Description'].apply(clean_text)

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(df['clean_text'])

# Targets
y_cat = df['Ticket Type']
y_pri = df['Ticket Priority']

# -------------------------------
# TRAIN MODEL
# -------------------------------
X_train, X_test, y_cat_train, y_cat_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

_, _, y_pri_train, y_pri_test = train_test_split(
    X, y_pri, test_size=0.2, random_state=42
)

model_cat = LogisticRegression(max_iter=500)
model_pri = LogisticRegression(max_iter=500)

model_cat.fit(X_train, y_cat_train)
model_pri.fit(X_train, y_pri_train)

# -------------------------------
# EVALUATION
# -------------------------------
cat_pred = model_cat.predict(X_test)
pri_pred = model_pri.predict(X_test)

cat_acc = accuracy_score(y_cat_test, cat_pred)
pri_acc = accuracy_score(y_pri_test, pri_pred)

# -------------------------------
# DISPLAY METRICS
# -------------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

col1.metric("Category Accuracy", f"{round(cat_acc*100,2)}%")
col2.metric("Priority Accuracy", f"{round(pri_acc*100,2)}%")

# -------------------------------
# USER INPUT
# -------------------------------
st.subheader("🧠 Predict New Ticket")

user_input = st.text_area("Enter Support Ticket Description")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        clean_input = clean_text(user_input)
        vector_input = vectorizer.transform([clean_input])

        pred_cat = model_cat.predict(vector_input)[0]
        pred_pri = model_pri.predict(vector_input)[0]

        st.success(f"📂 Category: {pred_cat}")
        st.info(f"⚡ Priority: {pred_pri}")

# -------------------------------
# SHOW SAMPLE DATA
# -------------------------------
st.subheader("📄 Sample Data")
st.dataframe(df.head())