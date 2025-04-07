import streamlit as st
from core.recommender import ProductRecommender

# App Config (must be the first Streamlit command)
st.set_page_config(
    page_title="SHL Assessment Recommender Pro",
    layout="centered",
    page_icon="📊"
)

# Initialize with caching
@st.cache_resource
def load_recommender():
    return ProductRecommender()

recommender = load_recommender()

# Custom CSS
st.markdown("""
<style>
    .stTextInput > div > div > input { font-size: 16px; padding: 12px; }
    .recommendation-card {
        border-left: 4px solid #2e86de;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-radius: 0 8px 8px 0;
        background: #f8f9fa;
    }
    .highlight { background-color: #fffacd; padding: 0.1rem 0.3rem; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 SHL Assessment Recommender Pro")
st.markdown("""
<div style='margin-bottom: 2rem;'>
    <p>AI-powered recommendations for SHL's assessment portfolio</p>
</div>
""", unsafe_allow_html=True)

# Input Section
with st.form("recommendation_form"):
    job_text = st.text_area(
        "🔍 Enter Job Description or Role:",
        placeholder="e.g. 'Financial Analyst with risk assessment skills'",
        height=120
    )
    
    col1, col2 = st.columns(2)
    with col1:
        experience_level = st.selectbox(
            "💼 Experience Level",
            ["Any", "Entry-Level", "Graduate", "Manager", "Director", "Executive"],
            index=0
        )
    with col2:
        categories = st.multiselect(
            "📂 Preferred Categories",
            ["Cognitive", "Personality", "Situational Judgment", "Technical"],
            default=["Cognitive", "Personality"]
        )
    
    submitted = st.form_submit_button("Get Recommendations")

# Results Section
if submitted:
    if not job_text.strip():
        st.warning("Please enter a job description")
    else:
        with st.spinner("Analyzing job requirements..."):
            try:
                results = recommender.recommend(
                    job_text,
                    None if experience_level == "Any" else experience_level,
                    categories if categories else None
                )
                
                if results.empty:
                    st.warning("No strong matches found. Showing general recommendations:")
                    results = recommender.df.sample(5).assign(score=0.4)
                
                st.subheader(f"🎯 Recommended Assessments ({len(results)} found)")
                
                for _, row in results.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class='recommendation-card'>
                            <h4>{row['Product']}</h4>
                            <p>{row['Description']}</p>
                            <div style='margin-top: 0.8rem;'>
                                <span style='color: #2e86de; font-weight: bold;'>Match: {row['score']:.0%}</span> | 
                                ⏱️ {row['Assessment Length (minutes)']} min | 
                                👥 {row['Job Levels']} | 
                                🗂️ {row['Category']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Download option
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Recommendations",
                    data=csv,
                    file_name="shl_recommendations.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Info Section
st.markdown("---")
st.markdown("""
### ℹ️ How It Works
1. **Semantic Matching**: Uses AI to understand job requirements  
2. **Smart Filters**: Applies experience level and category preferences  
3. **Fallback Logic**: Ensures recommendations even for niche roles  
""")
