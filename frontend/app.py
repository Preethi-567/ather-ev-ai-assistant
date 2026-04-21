import streamlit as st
import uuid
import sys
from pathlib import Path

# Fix module paths
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from src.frontend.api_client import send_query, send_feedback, fetch_analytics

# --- ENHANCED UI CONFIGURATION ---
st.set_page_config(
    page_title="Ather EV Assistant",
    page_icon="⚡",
    layout="wide"
)

# Custom Ather-inspired Styling
st.markdown("""
    <style>
    /* Global app background */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: linear-gradient(135deg, #0B1220, #111827) !important;
        color: #E5E7EB !important;
        font-family: 'Inter', sans-serif;
    }

    /* Main content area */
    .main,
    .block-container {
        background: transparent !important;
        color: #E5E7EB !important;
    }

    /* All normal text */
    p, span, label, div, li {
        color: #E5E7EB;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #F9FAFB !important;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B3D20, #0A1F14) !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #6EE7B7 !important;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    /* Chat message containers */
    .stChatMessage,
    [data-testid="stChatMessage"] {
        border-radius: 16px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.18) !important;
    }

    /* Assistant message */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background: rgba(255, 255, 255, 0.06) !important;
        border-left: 5px solid #22C55E !important;
    }

    /* User message */
    [data-testid="stChatMessage"]:nth-child(even) {
        background: rgba(34, 197, 94, 0.10) !important;
        border-left: 5px solid #F97316 !important;
    }

    /* Chat input / text input areas */
    [data-testid="stChatInput"] {
        background: transparent !important;
    }

    [data-testid="stChatInput"] textarea,
    textarea,
    input {
        background: rgba(255,255,255,0.08) !important;
        color: #F9FAFB !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: #C7D2FE !important;
        opacity: 0.8 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #16A34A, #22C55E) !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.6rem 1.25rem !important;
        font-weight: 600 !important;
        transition: all 0.25s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #15803D, #16A34A) !important;
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(34,197,94,0.25) !important;
    }

    /* Radio buttons / navigation labels */
    .stRadio label,
    .stRadio div {
        color: #F9FAFB !important;
    }

    /* Cards / metric / generic containers */
    [data-testid="stVerticalBlock"] > div:has(.element-container),
    [data-testid="stMarkdownContainer"],
    [data-testid="stText"] {
        color: #E5E7EB !important;
    }

    /* Optional source tag style */
    .source-tag {
        background: rgba(34, 197, 94, 0.12);
        color: #86EFAC;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        border: 1px solid rgba(34,197,94,0.25);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-thumb {
        background: #22C55E;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("# ⚡ ATHER")
    st.caption("Ather Owner's Digital Manual Assistant")
    st.divider()
    menu = st.radio("Navigation", ["Chat Assistant", "System Analytics"])
    st.divider()
    st.info("I use a hybrid search RAG pipeline to answer questions directly from official Ather manuals.")

# --- MAIN CONTENT ---
if menu == "Chat Assistant":
    st.title("Ather Assistant")
    
    # Display Chat History
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("pages"):
                st.markdown(f"<span class='source-tag'>Cited Pages: {', '.join(map(str, msg['pages']))}</span>", unsafe_allow_html=True)

    # Input Area
    if prompt := st.chat_input("Ask about Warp mode, charging, or troubleshooting..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Consulting manuals..."):
            result = send_query(prompt, st.session_state.session_id)
            
        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            
            # Show Cited Pages as pill-shaped tags
            if result.get("cited_pages"):
                st.markdown(f"<span class='source-tag'>Sources: {', '.join(map(str, result['cited_pages']))}</span>", unsafe_allow_html=True)

            # Feedback Row
            f_col1, f_col2, _ = st.columns([0.05, 0.05, 0.9])
            with f_col1:
                if st.button("👍", key=f"up_{result['row_id']}"):
                    send_feedback(result['row_id'], 1)
                    st.toast("Thank you!")
            with f_col2:
                if st.button("👎", key=f"down_{result['row_id']}"):
                    send_feedback(result['row_id'], -1)
                    st.toast("Feedback logged for review.")

        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["answer"],
            "pages": result.get("cited_pages", [])
        })

else:
    # --- ANALYTICS PAGE ---
    st.title("Admin Analytics Dashboard")
    stats = fetch_analytics()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Queries", stats["total_queries"])
    m2.metric("Positive Ratings", stats["thumbs_up"])
    m3.metric("Avg Latency", f"{stats['avg_latency_ms']}ms")

    st.subheader("Recent Activity Log")
    st.table(stats["recent"])