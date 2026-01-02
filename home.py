import streamlit as st

st.set_page_config(
    page_title="Real Estate CampusX",
    page_icon="🏠",
    layout="wide"
)

st.title("🏡 Real Estate Price Prediction System")

st.markdown("""
Welcome to **Real Estate CampusX**, an end-to-end machine learning application.

### 🔍 What this app does:
- Predicts property prices using ML pipelines
- Provides detailed market analysis
- Recommends similar properties using cosine similarity

### 📌 How to use:
Use the **sidebar** to navigate between different modules:
- **Price Predictor**
- **Analysis Dashboard**
- **Recommendation Engine**
""")

st.divider()

st.success("✅ Application deployed successfully on Streamlit Cloud")

st.sidebar.info("""
👨‍💻 Developed by **Shubham Dubey**  
🎯 AI / Data Science Project  
🚀 Streamlit + Machine Learning
""")
