
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Streamlit UI
st.title("ML Solution UI for Classification")
st.write("This is a demonstration of an ML-based solution.")

# Dataset Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format):")
if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)
    st.write("Preview of your data:")
    st.write(user_data)

# Classification
if st.button("Classify with Random Forest"):
    sample_data = X_test[:5]  # Use test samples for demonstration
    predictions = rf.predict(sample_data)
    st.write("Predictions for sample data:")
    st.write(predictions)
