# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("Employee Retention Analysis & Prediction")

# ==========================================
# 1. Load Dataset Safely
# ==========================================
# Robust fallback paths to prevent breaking on Streamlit Cloud
possible_paths = ["HR_comma_sep.csv", "/content/sample_data/HR_comma_sep.csv"]
df = None

for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        break

if df is None:
    st.error("Error: 'HR_comma_sep.csv' data file not found in your repository! Please upload it to your GitHub project folder.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================
st.subheader("Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    st.write("Dataset Description")
    st.write(df.describe())
with col2:
    st.write("Missing Values")
    st.write(df.isnull().sum())

# Correlation Heatmap
st.write("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
plt.title("Correlation Heatmap")
st.pyplot(fig)

# Employee Retention Count
st.write("Employee Retention Counts")
st.write(df['left'].value_counts())

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x='left', data=df, ax=ax)
plt.title("Employee Retention")
st.pyplot(fig)

# Compare averages
st.write("Averages Grouped by Target Variable ('left')")
st.write(df.groupby('left').mean(numeric_only=True))

# Bar Chart: Salary vs Retention
st.write("Salary vs Employee Retention")
salary_retention = pd.crosstab(df.salary, df.left)
fig, ax = plt.subplots(figsize=(7,5))
salary_retention.plot(kind='bar', ax=ax)
plt.title("Salary vs Employee Retention")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")
st.pyplot(fig)

# Bar Chart: Department vs Retention
st.write("Department vs Employee Retention")
dept_retention = pd.crosstab(df.Department, df.left)
fig, ax = plt.subplots(figsize=(12,6))
dept_retention.plot(kind='bar', ax=ax)
plt.title("Department vs Employee Retention")
plt.xlabel("Department")
plt.ylabel("Employees")
st.pyplot(fig)

# ==========================================
# 4. Build Logistic Regression Model
# ==========================================
X = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
salary_dummies = pd.get_dummies(X['salary'], prefix='salary', drop_first=True)
X = pd.concat([X.drop('salary', axis=1), salary_dummies], axis=1)
y = df['left']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ==========================================
# 5. Measure Accuracy
# ==========================================
st.subheader("Model Metrics")

accuracy = accuracy_score(y_test, y_pred)
st.metric(label="Accuracy Score", value=f"{accuracy:.4f}")

st.write("Confusion Matrix Dataframe:")
cm = confusion_matrix(y_test, y_pred)
st.write(cm)

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
st.pyplot(fig)
