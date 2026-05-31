import streamlit as st

st.set_page_config(
    page_title="EliteEstate Hub",
    page_icon="🏠",
    layout="wide"
)

# Custom Styling for modern UI cards
st.markdown("""
    <style>
        .welcome-title {
            font-size: 42px;
            font-weight: 700;
            color: #1e293b;
            font-family: 'Segoe UI', sans-serif;
            margin-bottom: 10px;
        }
        .feature-card {
            background-color: #ffffff;
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            height: 100%;
            margin-bottom: 20px;
        }
        .feature-icon {
            font-size: 32px;
            margin-bottom: 12px;
        }
        .feature-title {
            font-size: 20px;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 8px;
        }
        .feature-desc {
            font-size: 14px;
            color: #64748b;
            line-height: 1.5;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<div class="welcome-title">🏡 EliteEstate Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown("Welcome to an end-to-end, data-driven machine learning ecosystem built for smart property valuation and discovery.")

st.divider()

# High-Level System Summary Metrics
st.markdown("### 📊 System Overview")
m1, m2, m3 = st.columns(3)
with m1:
    st.metric(label="Machine Learning Model", value="Random Forest/XGBoost", delta="Pipeline Ready")
with m2:
    st.metric(label="Engine Logic", value="Cosine Similarity", delta="3-Layer Weights")
with m3:
    st.metric(label="Deployment Status", value="Streamlit Cloud", delta="Active", delta_color="inverse")

st.markdown("---")

# Core Features Showcased in Dynamic Grid Cards
st.markdown("### ⚡ Core Application Modules")
st.markdown("Use the left-hand sidebar menu to launch any of the specialized modules below:")

card_col1, card_col2, card_col3 = st.columns(3)

with card_col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Price Predictor</div>
        <div class="feature-desc">
            Input property attributes like location, square footage, room configurations, and furnishing status to estimate accurate market pricing valuations via advanced regression pipelines.
        </div>
    </div>
    """, unsafe_allow_html=True)

with card_col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Analysis Dashboard</div>
        <div class="feature-desc">
            Deep dive into geographic inventory density, price dispersion spreads via interactive charts, sunburst distributions, and regional keyword feature clouds.
        </div>
    </div>
    """, unsafe_allow_html=True)

with card_col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">Recommendation Engine</div>
        <div class="feature-desc">
            Discover sister properties using multi-matrix spatial indexing or locate physical assets situated within custom geographical distance parameters.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Footer Setup in Sidebar
st.sidebar.info("""
👨‍💻 **Developed by Shubham Dubey** 🎯 AI / Data Science Portfolio  
🚀 Streamlit + ML MLOps Structure
""")


# 1. Add a divider line in the sidebar
st.sidebar.markdown("---")

# 2. Add your developer title
st.sidebar.markdown("### 👨‍💻 Developer Profile")
st.sidebar.info("""
**Shubham Dubey** 🎯 AI / Data Science Specialist
""")

# 3. Add the clickable portfolio button (REPLACE THE URL BELOW WITH YOUR LINK)
st.sidebar.link_button(
    label="🌐 Visit My Portfolio",
    url="https://dazzling-pudding-0b3156.netlify.app/",
    use_container_width=True
)