import streamlit as st
import PIL.Image as Image
from pathlib import Path
import io

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# Set page config
st.set_page_config(page_title="Fracture AI", page_icon="🦴", layout="centered", initial_sidebar_state="expanded")

# Advanced CSS for a stunning, premium UI, focused on User Guidelines
st.markdown("""
<style>
    /* Global Theme Overrides */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }

    /* Top Padding adjustment */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Hero Section / Title */
    .hero-title {
        font-size: 3.5rem !important;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        padding-bottom: 0px;
        letter-spacing: -1px;
    }
    
    .hero-subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #94a3b8;
        font-weight: 300;
        margin-top: -10px;
        margin-bottom: 30px;
        letter-spacing: 0.5px;
    }

    /* Glassmorphism Container */
    .glass-container {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }

    /* Uploader Styling overriding */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(56, 189, 248, 0.05) !important;
        border: 2px dashed rgba(56, 189, 248, 0.4) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(56, 189, 248, 0.1) !important;
        border-color: #38bdf8 !important;
    }

    /* Beautiful Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding: 0.8rem 2rem !important;
        border: none !important;
        border-radius: 50px !important;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4), 0 4px 6px -2px rgba(59, 130, 246, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: block !important;
        width: 100% !important;
        margin-top: 15px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 20px 25px -5px rgba(59, 130, 246, 0.5), 0 10px 10px -5px rgba(59, 130, 246, 0.2) !important;
    }
    .stButton > button:active {
        transform: translateY(1px) scale(0.98) !important;
    }

    /* Result Cards */
    .status-card {
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        margin: 20px 0;
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .status-danger {
        background: radial-gradient(circle at top right, rgba(239, 68, 68, 0.15), rgba(153, 27, 27, 0.4));
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.15);
    }
    
    .status-safe {
        background: radial-gradient(circle at top right, rgba(34, 197, 94, 0.15), rgba(20, 83, 45, 0.4));
        border: 1px solid rgba(34, 197, 94, 0.3);
        box-shadow: 0 0 40px rgba(34, 197, 94, 0.15);
    }

    .status-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .status-danger .status-title { color: #fca5a5; }
    .status-safe .status-title { color: #86efac; }
    
    .status-sub {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Hide standard st elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    @keyframes slideUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Image display container */
    .img-showcase {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }
    
    /* Guidelines box styling for sidebar */
    .guideline-box {
        background: rgba(56, 189, 248, 0.05);
        border-left: 4px solid #38bdf8;
        padding: 15px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 15px;
    }
    .guideline-title {
        color: #38bdf8;
        font-weight: 600;
        margin-bottom: 5px;
        font-size: 1.1rem;
    }
    .guideline-text {
        font-size: 0.9rem;
        color: #cfd8dc;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- UI Structure -----------------

st.markdown("<h1 class='hero-title'>FractureVision AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Medical-grade AI for bone fracture detection</p>", unsafe_allow_html=True)

# Wrap main content in a glass container
st.markdown("<div class='glass-container'>", unsafe_allow_html=True)

# Sidebar - Focused entirely on beautifully presented user guidelines
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063229.png", width=80)
    st.markdown("<h1 style='color: white; font-size: 1.8rem; margin-top: 10px;'>How to Use</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    <div class='guideline-box'>
        <div class='guideline-title'>📸 1. Upload X-Ray</div>
        <div class='guideline-text'>Drag and drop or select a clear, high-quality radiograph image. We support <b>.jpg</b>, <b>.jpeg</b>, and <b>.png</b> formats.</div>
    </div>
    
    <div class='guideline-box'>
        <div class='guideline-title'>🚀 2. Analyze</div>
        <div class='guideline-text'>Click the glowing blue <b>INITIATE SCAN</b> button. Our AI will scan the entire bone structure for anomalies.</div>
    </div>
    
    <div class='guideline-box'>
        <div class='guideline-title'>🩺 3. Review Results</div>
        <div class='guideline-text'>The AI will highlight any suspected fracture zones. Review the <b>Diagnostic View</b> carefully.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("💡 **Tip:** For the most accurate results, ensure the uploaded X-ray image is well-lit and the fracture area (if any) is fully visible.")


# Locate model silently
model_path = "best (1).pt"
model_files = list(Path(".").glob("best*.pt"))
if len(model_files) > 0:
    model_path =  str(max(model_files, key=lambda p: p.stat().st_mtime))

# Loading the model silently
@st.cache_resource(show_spinner="Preparing analysis tools...")
def load_model():
    if not Path(model_path).exists():
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception:
        return None

if HAS_YOLO:
    model = load_model()
    
    if model:
        # File uploader
        uploaded_file = st.file_uploader("Drop radiograph here (.png, .jpg)", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # We have a file!
            image = Image.open(uploaded_file)
            col_img, col_btn = st.columns([3, 2])
            
            with col_img:
                st.markdown("<div style='border-radius:10px; overflow:hidden; border:1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
                st.image(image, caption="Source Image Ready", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_btn:
                st.markdown("<div style='height: 20%;'></div>", unsafe_allow_html=True) # spacer
                st.markdown("<h3 style='color:#e2e8f0; font-weight:300;'>Start Processing</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>Click below to begin analyzing the radiograph for fractures.</p>", unsafe_allow_html=True)
                analyze_btn = st.button("🚀 INITIATE SCAN")

            # Perform Inference
            if analyze_btn or st.session_state.get('last_uploaded') == uploaded_file.name:
                st.session_state['last_uploaded'] = uploaded_file.name
                
                st.markdown("---")
                
                with st.spinner("Analyzing radiograph..."):
                    try:
                        results = model(image)
                        res_plotted = results[0].plot(line_width=3)
                        boxes = results[0].boxes
                    except Exception as e:
                         st.error(f"⚠️ Something went wrong while analyzing this image. Please try a different X-ray file. (Error: {e})")
                         st.stop()  # halts execution here so the rest of the code (status card, etc.) doesn't run on bad data
                        
                    # Display the big status card
                    if len(boxes) > 0:
                        st.markdown(f"""
                        <div class='status-card status-danger'>
                            <div class='status-title'>⚠️ ABNORMALITY DETECTED</div>
                            <div class='status-sub' style='color:#fca5a5;'>Identified <b>{len(boxes)}</b> potential fracture zone(s). Requires medical review.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class='status-card status-safe'>
                            <div class='status-title'>✅ NO FRACTURE DETECTED</div>
                            <div class='status-sub' style='color:#86efac;'>No fractures were identified in this radiograph.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show Result Image
                    st.markdown("<h3 style='text-align:center; color:#e2e8f0; margin-top:20px; text-transform:uppercase; letter-spacing:2px; font-size:1.2rem;'>Diagnostic View</h3>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='img-showcase'>", unsafe_allow_html=True)
                    st.image(res_plotted, channels="BGR", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
    else:
        st.error("Model could not be loaded. Please ensure `best.pt` exists in the folder.")
else:
    st.error("Please install dependencies: `pip install ultralytics`")

st.markdown("</div>", unsafe_allow_html=True) # End glass container
