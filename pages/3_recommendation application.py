import pickle
import streamlit as st
import pandas as pd

st.set_page_config(page_title='EliteEstate | Recommendation System', layout='wide')
st.title('🤖 Intelligent Matchmaking & Recommendation Portal')
st.markdown(
    "Locate nearby properties within a specified radius or discover alternative properties using mathematical similarity engines.")
st.markdown("---")

# Import asset layers
location_df = pickle.load(open('location_distance.pkl', 'rb'))
cosine_sim1 = pickle.load(open('cosine_sim1.pkl', 'rb'))
cosine_sim2 = pickle.load(open('cosine_sim2.pkl', 'rb'))
cosine_sim3 = pickle.load(open('cosine_sim3.pkl', 'rb'))
df1 = pickle.load(open('df1.pkl', 'rb'))


def recommend_properties_with_scores(property_name, top_n=5):
    # Core mathematical formula matching your pipeline
    cosine_sin_matrix = 30 * cosine_sim1 + 20 * cosine_sim2 + 8 * cosine_sim3
    sim_scores = list(enumerate(cosine_sin_matrix[location_df.index.get_loc(property_name)]))
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
    top_properties = location_df.index[top_indices].tolist()

    return pd.DataFrame({'Alternative Property': top_properties, 'Match Confidence Score': top_scores})


# Split the engine UI neatly into two distinct interactive columns
left_pane, right_pane = st.columns(2)

with left_pane:
    st.markdown("### 📍 Geo-Distance Explorer")
    st.caption("Find community options situated around an explicit distance boundary radius.")

    ui_loc = st.selectbox('Anchor Point Location', sorted(location_df.columns.to_list()))
    ui_rad = st.number_input('Maximum Search Distance Radius (Kms)', min_value=0.5, value=5.0, step=0.5)

    if st.button('🚀 Scan Proximity Perimeter', use_container_width=True):
        selected_location = location_df[location_df[ui_loc] < ui_rad * 1000][ui_loc].sort_values()

        if not selected_location.empty:
            st.write(f"**Properties located within {ui_rad} km boundary:**")

            # Format the output into a clean, modern DataFrame view instead of raw printed text strings
            geo_results = []
            for key, value in selected_location.items():
                geo_results.append({"Property": key, "Distance": f"{round(value / 1000, 2)} Kms"})
            st.dataframe(pd.DataFrame(geo_results), use_container_width=True, hide_index=True)
        else:
            st.warning("No residential structures tracked within that precise perimeter radius.")

with right_pane:
    st.markdown("### 🧠 Content Similarity Matcher")
    st.caption("Discover matching alternatives calculated using similarity scoring indices.")

    selected_apartment = st.selectbox('Target Property Profile Baseline', df1['PropertyName'].unique())

    if st.button('🔍 Generate Similar Alternatives', use_container_width=True):
        rec_df = recommend_properties_with_scores(selected_apartment)
        st.write("**Top 5 Recommended Alternatives:**")
        st.dataframe(rec_df, use_container_width=True, hide_index=True)


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