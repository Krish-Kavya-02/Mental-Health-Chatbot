import streamlit as st
import time
from datetime import datetime
from ragpipeline import create_rag_pipeline
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ───
st.set_page_config(
    page_title="MindMate — Mental Health Support",
    page_icon="🧠",
    layout="centered"
)

# ─── Session State ───
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_suggestion" not in st.session_state:
    st.session_state.pending_suggestion = None
if "last_followups" not in st.session_state:
    st.session_state.last_followups = []

# ─── Load Pipeline with animated spinner ───
@st.cache_resource(show_spinner=False)
def get_chain():
    return create_rag_pipeline()

# Show custom loading animation if not cached
if "pipeline_loaded" not in st.session_state:
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
    <div class="init-screen">
        <div class="init-brain-container">
            <div class="init-pulse-ring"></div>
            <div class="init-pulse-ring delay-1"></div>
            <div class="init-pulse-ring delay-2"></div>
            <div class="init-brain">🧠</div>
        </div>
        <div class="init-title">MindMate</div>
        <div class="init-subtitle">Preparing your mental health companion</div>
        <div class="init-loader">
            <div class="init-bar"></div>
        </div>
        <div class="init-tips">
            <span class="init-tip-item">Loading knowledge base</span>
            <span class="init-dot">·</span>
            <span class="init-tip-item">Indexing resources</span>
            <span class="init-dot">·</span>
            <span class="init-tip-item">Almost ready</span>
        </div>
    </div>
    <style>
    .init-screen {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 80vh;
        animation: initFadeIn 0.6s ease;
    }
    @keyframes initFadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .init-brain-container {
        position: relative;
        width: 120px;
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 32px;
    }
    .init-brain {
        font-size: 56px;
        z-index: 2;
        animation: initFloat 3s ease-in-out infinite;
    }
    @keyframes initFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .init-pulse-ring {
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        border: 2px solid rgba(99, 102, 241, 0.3);
        animation: initPulse 2s ease-out infinite;
    }
    .delay-1 { animation-delay: 0.4s; }
    .delay-2 { animation-delay: 0.8s; }
    @keyframes initPulse {
        0% { transform: scale(0.8); opacity: 1; }
        100% { transform: scale(2); opacity: 0; }
    }
    .init-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.03em;
        margin-bottom: 8px;
    }
    .init-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 36px;
    }
    .init-loader {
        width: 200px;
        height: 3px;
        background: rgba(255,255,255,0.06);
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 24px;
    }
    .init-bar {
        width: 40%;
        height: 100%;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #6366f1);
        border-radius: 4px;
        animation: initSlide 1.5s ease-in-out infinite;
    }
    @keyframes initSlide { 0% { transform: translateX(-100%); } 100% { transform: translateX(350%); } }
    .init-tips {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    .init-tip-item { color: #475569; font-size: 0.8rem; animation: initBlink 2s ease-in-out infinite; }
    .init-dot { color: #334155; }
    @keyframes initBlink { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }
    </style>
    """, unsafe_allow_html=True)
    
    ask_with_followups = get_chain()
    st.session_state.pipeline_loaded = True
    loading_placeholder.empty()
    st.rerun()
else:
    ask_with_followups = get_chain()

# ─── Get current time for messages ───
def get_time():
    return datetime.now().strftime("%I:%M %p")

# ─── Full UI Styling ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ─── Global ─── */
    .stApp {
        background: #08080d;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 20% 20%, rgba(99,102,241,0.04) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(139,92,246,0.03) 0%, transparent 50%);
        animation: bgShift 20s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    @keyframes bgShift {
        0%, 100% { transform: translate(0, 0); }
        33% { transform: translate(-2%, 1%); }
        66% { transform: translate(1%, -2%); }
    }
    
    /* ─── Hide defaults ─── */
    .stApp header, #MainMenu, footer, 
    h1[data-testid="stHeading"] { display: none !important; visibility: hidden !important; }
    [data-testid="stDeployButton"] { display: none !important; }
    
    /* ─── Header ─── */
    .mm-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 28px 0 20px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        margin-bottom: 24px;
        animation: slideDown 0.5s ease;
    }
    @keyframes slideDown { from { opacity: 0; transform: translateY(-16px); } to { opacity: 1; transform: translateY(0); } }
    .mm-logo {
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 24px;
        box-shadow: 0 4px 20px rgba(99,102,241,0.25);
        animation: logoPulse 3s ease-in-out infinite;
    }
    @keyframes logoPulse {
        0%, 100% { box-shadow: 0 4px 20px rgba(99,102,241,0.25); }
        50% { box-shadow: 0 4px 30px rgba(99,102,241,0.4); }
    }
    .mm-title { font-size: 1.5rem; font-weight: 700; color: #f1f5f9; letter-spacing: -0.03em; }
    .mm-tag { font-size: 0.85rem; color: #6366f1; font-weight: 500; letter-spacing: 0.01em; }
    .mm-status { display: flex; align-items: center; gap: 6px; margin-left: auto; }
    .mm-dot { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; animation: dotPulse 2s infinite; }
    @keyframes dotPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .mm-status-text { font-size: 0.75rem; color: #4ade80; font-weight: 500; }
    
    /* ─── Chat Messages ─── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 12px 4px !important;
        margin-bottom: 4px !important;
        animation: msgSlideIn 0.4s ease;
    }
    @keyframes msgSlideIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
    
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: #d1d5db;
        font-size: 1.02rem;
        line-height: 1.8;
    }
    [data-testid="stChatMessage"] strong {
        color: #a5b4fc;
        font-weight: 600;
    }
    [data-testid="stChatMessage"] hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 16px 0 !important;
    }
    
    /* ─── Chat Input ─── */
    [data-testid="stChatInput"] {
        border-top: 1px solid rgba(255,255,255,0.04);
        padding-top: 8px;
    }
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        padding: 14px 18px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(99,102,241,0.4) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.08) !important;
        background: rgba(255,255,255,0.04) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #4b5563 !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stChatInput"] button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important;
    }
    
    /* ─── Suggestion Buttons ─── */
    .stButton > button {
        background: rgba(99,102,241,0.06) !important;
        border: 1px solid rgba(99,102,241,0.12) !important;
        border-radius: 12px !important;
        color: #818cf8 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 12px 16px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Inter', sans-serif !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 48px !important;
        line-height: 1.5 !important;
    }
    .stButton > button:hover {
        background: rgba(99,102,241,0.15) !important;
        border-color: rgba(99,102,241,0.35) !important;
        color: #c7d2fe !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(99,102,241,0.15) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* ─── Timestamps ─── */
    .msg-time {
        font-size: 0.7rem;
        color: #374151;
        margin-top: 4px;
        font-weight: 400;
    }
    
    /* ─── Captions ─── */
    [data-testid="stCaptionContainer"] p {
        color: #2d3748 !important;
        font-size: 0.72rem !important;
    }
    
    /* ─── Welcome Screen ─── */
    .welcome-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 55vh;
        animation: welcomeFadeUp 0.8s ease;
        padding: 20px 0;
    }
    @keyframes welcomeFadeUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
    
    .welcome-glow {
        position: relative;
        margin-bottom: 28px;
    }
    .welcome-glow::before {
        content: '';
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        width: 140px; height: 140px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
        animation: glowPulse 3s ease-in-out infinite;
    }
    @keyframes glowPulse { 0%, 100% { transform: translate(-50%,-50%) scale(1); opacity: 1; } 50% { transform: translate(-50%,-50%) scale(1.3); opacity: 0.6; } }
    .welcome-icon {
        font-size: 64px;
        position: relative;
        z-index: 1;
        animation: floatIcon 4s ease-in-out infinite;
    }
    @keyframes floatIcon {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        25% { transform: translateY(-6px) rotate(2deg); }
        75% { transform: translateY(4px) rotate(-2deg); }
    }
    
    .welcome-heading {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.03em;
        margin-bottom: 12px;
        text-align: center;
    }
    .welcome-desc {
        color: #64748b;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 480px;
        text-align: center;
        margin-bottom: 36px;
    }
    
    .welcome-features {
        display: flex;
        gap: 20px;
        flex-wrap: wrap;
        justify-content: center;
        margin-bottom: 40px;
        max-width: 540px;
    }
    .feature-item {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #94a3b8;
        font-size: 0.9rem;
        animation: featureFadeIn 0.6s ease both;
    }
    .feature-item:nth-child(1) { animation-delay: 0.2s; }
    .feature-item:nth-child(2) { animation-delay: 0.35s; }
    .feature-item:nth-child(3) { animation-delay: 0.5s; }
    .feature-item:nth-child(4) { animation-delay: 0.65s; }
    @keyframes featureFadeIn { from { opacity: 0; transform: translateX(-10px); } to { opacity: 1; transform: translateX(0); } }
    .feature-icon {
        width: 32px; height: 32px;
        background: rgba(99,102,241,0.08);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }
    
    /* ─── Section labels ─── */
    .section-label {
        color: #475569;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 16px 0 10px 0;
    }
    
    /* ─── Footer disclaimer ─── */
    .footer-note {
        text-align: center;
        color: #334155;
        font-size: 0.72rem;
        padding: 16px 20px 8px;
        border-top: 1px solid rgba(255,255,255,0.03);
        margin-top: 16px;
        line-height: 1.6;
    }
    
    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.15); border-radius: 4px; }
    
    /* ─── Dividers ─── */
    hr { border-color: rgba(255,255,255,0.04) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown("""
<div class="mm-header">
    <div class="mm-logo">🧠</div>
    <div>
        <div class="mm-title">MindMate</div>
        <div class="mm-tag">Mental Health Support</div>
    </div>
    <div class="mm-status">
        <div class="mm-dot"></div>
        <span class="mm-status-text">Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Process Pending Suggestion ───
if st.session_state.pending_suggestion is not None:
    suggestion_text = st.session_state.pending_suggestion
    st.session_state.pending_suggestion = None
    
    st.session_state.chat_history.append({
        "role": "user", "content": suggestion_text, "time": get_time()
    })
    
    history = [(e["role"], e["content"]) for e in st.session_state.chat_history]
    response = ask_with_followups(suggestion_text, history)
    answer = response.get("reply", "Could you tell me more about how you're feeling?")
    
    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "time": get_time()
    })
    st.session_state.last_followups = response.get("followups", [])
    st.rerun()

# ─── Welcome Screen ───
if not st.session_state.chat_history:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-glow">
            <div class="welcome-icon">🧠</div>
        </div>
        <div class="welcome-heading">How can I support you today?</div>
        <div class="welcome-desc">
            I'm your mental health companion — here to listen, understand, and share helpful resources. 
            Everything you share is private and confidential.
        </div>
        <div class="welcome-features">
            <div class="feature-item"><div class="feature-icon">🔒</div> Private & Confidential</div>
            <div class="feature-item"><div class="feature-icon">📚</div> Evidence-Based Resources</div>
            <div class="feature-item"><div class="feature-icon">⚡</div> Instant Responses</div>
            <div class="feature-item"><div class="feature-icon">🤝</div> Non-Judgmental Support</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-label">Start a conversation</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    starters = [
        "I've been feeling anxious lately",
        "I'm having trouble sleeping",
        "I feel overwhelmed and stressed"
    ]
    for i, text in enumerate(starters):
        with cols[i]:
            if st.button(text, key=f"starter_{i}", use_container_width=True):
                st.session_state.pending_suggestion = text
                st.rerun()
    
    st.markdown("""
    <div class="footer-note">
        MindMate is a supportive companion, not a replacement for professional care.<br>
        In crisis? Contact <strong>988 Suicide & Crisis Lifeline</strong> — call or text 988, available 24/7.
    </div>
    """, unsafe_allow_html=True)

# ─── Chat History ───
for idx, entry in enumerate(st.session_state.chat_history):
    role = entry["role"]
    content = entry["content"]
    msg_time = entry.get("time", "")
    avatar = "👤" if role == "user" else "🧠"
    
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        if msg_time:
            st.markdown(f'<div class="msg-time">{msg_time}</div>', unsafe_allow_html=True)

# ─── Follow-up Suggestions ───
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
    followups = st.session_state.last_followups
    if followups:
        st.markdown('<p class="section-label">Related topics</p>', unsafe_allow_html=True)
        cols = st.columns(min(len(followups), 3))
        for i, suggestion in enumerate(followups[:3]):
            with cols[i]:
                if st.button(suggestion, key=f"followup_{len(st.session_state.chat_history)}_{i}", use_container_width=True):
                    st.session_state.pending_suggestion = suggestion
                    st.rerun()

# ─── Chat Input ───
user_input = st.chat_input("Type your message...", key="main_input")

if user_input:
    current_time = get_time()
    st.session_state.chat_history.append({
        "role": "user", "content": user_input, "time": current_time
    })
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
        st.markdown(f'<div class="msg-time">{current_time}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner(""):
            history = [(e["role"], e["content"]) for e in st.session_state.chat_history]
            response = ask_with_followups(user_input, history)
        
        answer = response.get("reply", "Could you tell me more about how you're feeling?")
        resp_time = get_time()
        
        st.markdown(answer)
        st.markdown(f'<div class="msg-time">{resp_time}</div>', unsafe_allow_html=True)
    
    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "time": resp_time
    })
    st.session_state.last_followups = response.get("followups", [])
    st.rerun()
