#Import the libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set page configuration for a premium look
st.set_page_config(
    page_title="Employee Retention Insights",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to tweak styles
st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: 700; color: #1E3A8A; margin-bottom: 20px; }
    .section-desc { font-size: 16px; color: #4B5563; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. Load & Process Dataset Safely
# ==========================================
possible_paths = ["HR_comma_sep.csv", "/content/sample_data/HR_comma_sep.csv"]
df = None

for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        break

if df is None:
    st.error("🚨 Error: 'HR_comma_sep.csv' data file not found! Please ensure it is uploaded to your GitHub repository.")
    st.stop()

# Cache the data and model pipeline so it doesn't re-run unnecessarily
@st.cache_resource
def train_model(data):
    # Features used in your original logic
    X_features = data[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
    # Get dummies matching your exact training setup
    salary_dummies = pd.get_dummies(X_features['salary'], prefix='salary', drop_first=True)
    X_final = pd.concat([X_features.drop('salary', axis=1), salary_dummies], axis=1)
    y_final = data['left']
    
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
    
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    return lr_model, X_train.columns, X_test, y_test

model, training_columns, X_test, y_test = train_model(df)
y_pred = model.predict(X_test)

# ==========================================
# 2. Sidebar Navigation Layout
# ==========================================
st.sidebar.title("💼 Navigation")
st.sidebar.markdown("Explore the sections of the project below:")
app_mode = st.sidebar.radio(
    "Go To:",
    ["🏠 Home & Data Preview", "📊 Visual Analytics", "🤖 Retention Predictor", "📈 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Quick Stats")
st.sidebar.metric(label="Total Employee Records", value=len(df))
st.sidebar.metric(label="Model Framework", value="Logistic Regression")

# ==========================================
# Tab 1: Home & Data Preview
# ==========================================
if app_mode == "🏠 Home & Data Preview":
    st.markdown('<div class="main-title">Employee Retention Analysis & Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Welcome! This interactive dashboard analyzes HR demographics and uses a **Logistic Regression Machine Learning Model** to predict whether an employee is likely to stay or leave an organization based on core workplace metrics.
        """
    )
    
    st.subheader("🔍 Dataset Overview")
    st.markdown("Here is a slice of the raw HR analytics data used to train our prediction models:")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Structural Summary")
        st.write(df.describe())
    with col2:
        st.markdown("### 🔍 Missing Values Check")
        missing_data = df.isnull().sum().to_frame(name="Missing Count")
        st.dataframe(missing_data, use_container_width=True)

# ==========================================
# Tab 2: Visual Analytics
# ==========================================
elif app_mode == "📊 Visual Analytics":
    st.markdown('<div class="main-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.pills = ["Correlation Heatmap", "Retention Counts", "Department & Salary Splits"]
    
    # Grid Breakdown 1
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### 🌡️ Numeric Correlation Matrix")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    with c2:
        st.markdown("### ⚖️ Retention Distribution")
        fig, ax = plt.subplots(figsize=(5, 5.2))
        sns.countplot(x='left', data=df, palette="Set2", ax=ax)
        ax.set_xticklabels(["Stayed (0)", "Left (1)"])
        st.pyplot(fig)
        
        st.markdown("**Retention Value Counts:**")
        st.write(df['left'].value_counts())

    st.markdown("---")
    
    # Grid Breakdown 2
    st.markdown("### 🏢 Organizational Influences on Churn")
    tab_salary, tab_dept = st.tabs(["Salary Impact", "Department Impact"])
    
    with tab_salary:
        st.markdown("#### Salary vs Employee Retention")
        salary_retention = pd.crosstab(df.salary, df.left)
        fig, ax = plt.subplots(figsize=(8, 4))
        salary_retention.plot(kind='bar', stacked=True, ax=ax, color=['#2b5c8f', '#d95f02'])
        plt.ylabel("Number of Employees")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        
    with tab_dept:
        st.markdown("#### Department vs Employee Retention")
        dept_retention = pd.crosstab(df.Department, df.left)
        fig, ax = plt.subplots(figsize=(12, 5))
        dept_retention.plot(kind='bar', ax=ax)
        plt.ylabel("Employees")
        st.pyplot(fig)

    st.markdown("### 📋 Targeted Feature Group Averages")
    st.write(df.groupby('left').mean(numeric_only=True))

# ==========================================
# Tab 3: Retention Predictor (Updated for High-Risk Focus)
# ==========================================
elif app_mode == "🤖 Retention Predictor":
    st.markdown('<div class="main-title">Live Attrition Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown("Input an employee's profile below to see if the model flags them as likely to **LEAVE**.")
    
    with st.form("prediction_form"):
        col_left, col_right = st.columns(2)
        
        with col_left:
            satisfaction = st.slider("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.3, step=0.05, 
                                     help="0.0 = Miserable, 1.0 = Highly Satisfied")
            monthly_hours = st.number_input("Average Monthly Hours Worked", min_value=1, max_value=500, value=250)
            
        with col_right:
            salary_level = st.selectbox("Employee Salary Bracket", options=["low", "medium", "high"])
            promotion = st.selectbox("Promoted in Last 5 Years?", options=["No", "Yes"])
            promotion_val = 1 if promotion == "Yes" else 0
            
        submit_btn = st.form_submit_button("Run Departure Risk Assessment")
        
    if submit_btn:
        # Construct raw input features
        input_data = pd.DataFrame([{
            'satisfaction_level': satisfaction,
            'average_montly_hours': monthly_hours,
            'promotion_last_5years': promotion_val,
            'salary_low': 1 if salary_level == 'low' else 0,
            'salary_medium': 1 if salary_level == 'medium' else 0
        }])
        
        input_data = input_data.reindex(columns=training_columns, fill_value=0)
        
        prediction = model.predict(input_data)[0]
        probability_leave = model.predict_proba(input_data)[0][1] # Probability of leaving (Class 1)
        
        st.markdown("---")
        st.markdown("## 🔍 Assessment Verdict")
        
        # EXACTLY SAY WHEN AN EMPLOYEE WILL LEAVE:
        # If probability is greater than 50% (0.5), Logistic Regression flags it as a 1 (Leave)
        if prediction == 1 or probability_leave >= 0.5:
            st.error(f"🚨 **CRITICAL ATTRITION RISK DETECTED: THE EMPLOYEE IS PREDICTED TO LEAVE.**")
            
            # Display metrics card focusing on departure
            st.metric(label="Probability of Departure", value=f"{probability_leave:.1%}")
            
            st.markdown("### 📉 Core Triggers for this Departure Prediction:")
            st.markdown("Based on the mathematical coefficients of your Logistic Regression model, this employee crosses the line into 'Leaving' because:")
            if satisfaction < 0.4:
                st.markdown("* **Severely Low Satisfaction:** A satisfaction level below 0.40 is the strongest mathematical driver for employee turnover in this dataset.")
            if monthly_hours > 240:
                st.markdown("* **Overworked Hours:** The average monthly hours are dangerously high, pointing to severe burnout risk.")
            if salary_level == "low":
                st.markdown("* **Low Stagnant Salary:** Being stuck in a low salary bracket significantly multiplies the probability of departure.")
            if promotion_val == 0:
                st.markdown("* **No Recent Promotion:** Lack of upward mobility in the last 5 years acts as a compounding departure catalyst.")
                
        else:
            st.success(f"✅ **RETENTION PROFILE SAFE:** The employee is expected to **STAY**.")
            st.metric(label="Probability of Departure", value=f"{probability_leave:.1%}")
            st.markdown("The departure probability is safely below the **50% classification threshold**, meaning their metrics do not match the typical signature of someone planning to quit.")
# ==========================================
# Tab 4: Model Performance
# ==========================================
elif app_mode == "📈 Model Performance":
    st.markdown('<div class="main-title">Model Diagnostics & Metrics</div>', unsafe_allow_html=True)
    
    accuracy = accuracy_score(y_test, y_pred)
    st.metric(label="Overall Testing Accuracy Score", value=f"{accuracy:.2%}")
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("### 🧩 Confusion Matrix Visualization")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        st.pyplot(fig)
        
    with c_m2:
        st.markdown("### 📄 Classification Evaluation Report")
        st.text_area(label="Precision, Recall & F1-Score Breakdown", value=classification_report(y_test, y_pred), height=220)

# Sidebar Brading ------

st.markdown("---")

st.markdown(
"""
<div style="
padding:25px;
border-radius:15px;
background:linear-gradient(135deg,#0f172a,#1e293b);
color:white;
text-align:center;
">

<h2>👨‍💻 Developer</h2>

<h3>Abhay Kumar Gupta</h3>

<p>Machine Learning • Deep Learning • Python • Streamlit</p>

<a href="https://github.com/Abhay-cody" target="_blank">
<img src="https://img.shields.io/badge/GitHub-Visit_Profile-181717?style=for-the-badge&logo=github">
</a>

<br><br>

<a href="https://www.linkedin.com/in/abhay-kumar-gupta-104a18397" target="_blank">
<img src="https://img.shields.io/badge/LinkedIn-Connect_with_Me-0077B5?style=for-the-badge&logo=linkedin">
</a>

</div>
""",
unsafe_allow_html=True
)
