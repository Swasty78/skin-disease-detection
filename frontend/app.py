"""
Skin Disease Detection - Streamlit Frontend
A user-friendly interface for skin disease analysis using AI.
"""
import os
import streamlit as st
import httpx
from PIL import Image
import io
from datetime import datetime

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 60.0  # seconds

# Page configuration
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    .severity-benign {
        color: #28a745;
        font-weight: bold;
    }
    .severity-mild {
        color: #ffc107;
        font-weight: bold;
    }
    .severity-moderate {
        color: #fd7e14;
        font-weight: bold;
    }
    .severity-serious {
        color: #dc3545;
        font-weight: bold;
    }
    .confidence-high {
        color: #28a745;
    }
    .confidence-medium {
        color: #ffc107;
    }
    .confidence-low {
        color: #dc3545;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


def get_severity_class(severity: str) -> str:
    """Get CSS class for severity level."""
    severity_map = {
        "benign": "severity-benign",
        "mild": "severity-mild",
        "mild-moderate": "severity-moderate",
        "moderate": "severity-moderate",
        "serious": "severity-serious"
    }
    return severity_map.get(severity.lower(), "")


def get_confidence_class(confidence: float) -> str:
    """Get CSS class for confidence level."""
    if confidence >= 0.85:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    return "confidence-low"


def get_confidence_label(confidence: float) -> str:
    """Get human-readable confidence label."""
    if confidence >= 0.85:
        return "High Confidence"
    elif confidence >= 0.6:
        return "Moderate Confidence"
    return "Low Confidence"


@st.cache_data(ttl=60)
def fetch_diseases():
    """Fetch list of supported diseases from API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_URL}/diseases")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        st.warning(f"Could not fetch disease list: {e}")
    return None


@st.cache_data(ttl=30)
def fetch_history(limit: int = 10):
    """Fetch analysis history from API."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_URL}/history", params={"limit": limit})
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
    return []


def check_api_health():
    """Check if the API is healthy."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_URL}/health")
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
    return None


def analyze_image(image_bytes: bytes, filename: str) -> dict | None:
    """Send image to API for analysis."""
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            files = {"file": (filename, image_bytes, "image/jpeg")}
            response = client.post(f"{API_URL}/analyze_skin", files=files)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Analysis failed: {response.text}")
                return None
    except httpx.TimeoutException:
        st.error("Request timed out. The analysis is taking longer than expected.")
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 Skin Disease Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Skin Analysis with Personalized Recommendations</p>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for informational purposes only and is NOT a substitute 
        for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider 
        for proper evaluation of any skin condition.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Information")
        
        # API Status
        health = check_api_health()
        if health:
            st.success("✅ API Connected")
            with st.expander("System Status"):
                st.write(f"**Version:** {health.get('version', 'N/A')}")
                st.write(f"**Model Loaded:** {'✅' if health.get('model_loaded') else '❌'}")
                st.write(f"**LLM Loaded:** {'✅' if health.get('llm_loaded') else '❌'}")
                st.write(f"**Database:** {'✅' if health.get('database_connected') else '❌'}")
        else:
            st.error("❌ API Unavailable")
            st.info("Make sure the backend server is running.")
        
        st.divider()
        
        # Supported Diseases
        st.header("🦠 Supported Conditions")
        diseases_data = fetch_diseases()
        if diseases_data:
            for disease in diseases_data.get("diseases", []):
                severity = disease.get("severity", "unknown")
                severity_emoji = {
                    "benign": "🟢",
                    "mild": "🟡", 
                    "mild-moderate": "🟠",
                    "serious": "🔴"
                }.get(severity, "⚪")
                
                with st.expander(f"{severity_emoji} {disease['name']}"):
                    st.write(disease.get("description", ""))
                    st.caption(f"Severity: {severity} | Contagious: {'Yes' if disease.get('contagious') else 'No'}")
        
        st.divider()
        
        # History
        st.header("📜 Recent Analyses")
        history = fetch_history(limit=5)
        if history:
            for item in history:
                created = datetime.fromisoformat(item['created_at'].replace('Z', '+00:00'))
                st.write(f"**{item['disease']}** ({item['confidence']:.0%})")
                st.caption(created.strftime("%Y-%m-%d %H:%M"))
        else:
            st.info("No analysis history yet.")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a skin image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear, well-lit image of the affected skin area"
        )
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.caption(f"📁 {uploaded_file.name} | 📐 {image.size[0]}x{image.size[1]} px")
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image... This may take a few moments."):
                    # Reset file position
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    
                    result = analyze_image(image_bytes, uploaded_file.name)
                    
                    if result:
                        st.session_state['analysis_result'] = result
                        st.rerun()
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']
            
            # Disease and Confidence
            disease = result.get('disease', 'Unknown')
            confidence = result.get('confidence', 0)
            severity = result.get('severity', 'unknown')
            
            # Result card
            st.markdown(f"""
            <div class="result-card">
                <h2>{disease}</h2>
                <p class="{get_severity_class(severity)}">Severity: {severity.title()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence meter
            st.subheader("Confidence Level")
            confidence_label = get_confidence_label(confidence)
            st.progress(confidence, text=f"{confidence:.1%} - {confidence_label}")
            
            if confidence < 0.6:
                st.warning("⚠️ Low confidence prediction. Results may be unreliable.")
            
            # Recommendations
            st.divider()
            
            with st.expander("💊 Recommendations", expanded=True):
                st.write(result.get('recommendations', 'No recommendations available.'))
            
            with st.expander("👣 Next Steps", expanded=True):
                st.write(result.get('next_steps', 'No next steps available.'))
            
            with st.expander("💡 Tips & Advice", expanded=True):
                st.write(result.get('tips', 'No tips available.'))
            
            # Disclaimer in results
            st.info(result.get('disclaimer', 'Please consult a healthcare professional.'))
            
            # Clear results button
            if st.button("🗑️ Clear Results"):
                del st.session_state['analysis_result']
                st.rerun()
        else:
            st.info("👈 Upload an image and click 'Analyze' to see results here.")
            
            # Sample guidance
            with st.expander("📸 Tips for Best Results"):
                st.markdown("""
                For the most accurate analysis:
                - Use good lighting (natural light is best)
                - Take a clear, focused photo
                - Capture the affected area from about 6-12 inches away
                - Include some surrounding healthy skin for context
                - Avoid using filters or editing the image
                """)
    
    # Footer
    st.divider()
    st.caption(
        "Powered by AI | EfficientNet + LLM | "
        "For educational and informational purposes only"
    )


if __name__ == "__main__":
    main()
