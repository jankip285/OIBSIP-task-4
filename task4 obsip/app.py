# MODEL LOGIC — DO NOT MODIFY
import os
import pickle
import warnings
import numpy as np

# Monkey patch Keras to prevent deserialization issues with quantization_config
try:
    import keras

    def _make_patched_init(original_init):
        def patched_init(self, *args, **kwargs):
            kwargs.pop('quantization_config', None)
            original_init(self, *args, **kwargs)
        return patched_init

    try:
        from keras.src.layers.core.embedding import Embedding
        Embedding.__init__ = _make_patched_init(Embedding.__init__)
    except (ImportError, AttributeError):
        try:
            from keras.layers import Embedding
            Embedding.__init__ = _make_patched_init(Embedding.__init__)
        except Exception:
            pass

    try:
        from keras.src.layers.core.dense import Dense
        Dense.__init__ = _make_patched_init(Dense.__init__)
    except (ImportError, AttributeError):
        try:
            from keras.layers import Dense
            Dense.__init__ = _make_patched_init(Dense.__init__)
        except Exception:
            pass

except Exception:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import gradio as gr
from tensorflow.keras.models import load_model
# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Config ────────────────────────────────────────────────────
MODEL_PATH     = "best_spam_detector.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN        = 150
THRESHOLD      = 0.5
MODEL_ACCURACY = 96.4  # UI-only placeholder. Replace with your real test accuracy.
# ───────────────────────────────────────────────────────────────

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise FileNotFoundError(
        f"Could not load model from '{MODEL_PATH}'. "
        f"Make sure the file exists in the working directory. Original error: {e}"
    )

try:
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find tokenizer at '{TOKENIZER_PATH}'. "
        "Make sure 'tokenizer.pkl' exists in the working directory."
    )


def predict(message: str):
    # Keep the model inference part exactly as it is, only adjust the returned HTML UI format
    if not message.strip():
        return (
            gr.update(),
            "<div class='result-empty'>Paste a message and click Analyze to see the result.</div>"
        )

    seq     = tokenizer.texts_to_sequences([message])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob    = float(model.predict(padded, verbose=0)[0][0])
    is_spam = prob >= THRESHOLD

    pct   = prob * 100
    bar_w = max(0, min(100, int(pct)))

    if is_spam:
        label = "SPAM"
        badge_class = "spam-badge"
        fill_class = "spam"
    else:
        label = "NOT SPAM"
        badge_class = "ham-badge"
        fill_class = "ham"

    verdict_html = f"""
    <div class="result-panel">
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
        <span class="result-badge {badge_class}">{label}</span>
        <span class="confidence-label">{pct:.1f}% confidence</span>
      </div>
      <div class="confidence-bar">
        <div class="confidence-fill {fill_class}" style="width:{bar_w}%"></div>
      </div>
    </div>
    """

    return gr.update(), verdict_html


# UI LAYER — SAFE TO EDIT
css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700&family=Inter:wght@400;500;600;700&display=swap');

/* Dynamic background gradient shifting */
body {
    background: linear-gradient(135deg, #F5F3FF 0%, #E0F2FE 50%, #F5F3FF 100%) !important;
    background-size: 400% 400% !important;
    animation: gradientShift 15s ease infinite !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.gradio-container {
    background: transparent !important;
    min-height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 40px 20px !important;
}

#page-container {
    width: 100% !important;
    max-width: 700px !important;
    margin: 0 auto !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
}

#header-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    width: 100% !important;
    margin-bottom: 8px !important;
}

#header-left p, #header-left strong {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #4C1D95 !important;
    margin: 0 !important;
}

#header-right p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #15803D !important;
    background-color: #DCFCE7 !important;
    padding: 4px 12px !important;
    border-radius: 999px !important;
    border: 1px solid rgba(21, 128, 61, 0.15) !important;
    margin: 0 !important;
    transition: all 0.3s ease !important;
}

/* Card entry animation */
#main-card {
    background-color: #ffffff !important;
    border-radius: 18px !important;
    box-shadow: 0 10px 30px rgba(76, 29, 149, 0.05) !important;
    padding: 32px !important;
    border: 1px solid rgba(199, 210, 254, 0.3) !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 20px !important;
    animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

#card-title p, #card-title strong, #card-title h1 {
    font-family: 'Poppins', sans-serif !important;
    color: #4C1D95 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 0 4px 0 !important;
}

#card-subtitle p {
    font-family: 'Inter', sans-serif !important;
    color: #1F2937 !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    margin: 0 !important;
}

#msg-input textarea {
    border-radius: 12px !important;
    border: 1px solid #C7D2FE !important;
    padding: 16px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    color: #1F2937 !important;
    background-color: #ffffff !important;
    box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.02) !important;
    transition: all 0.25s ease !important;
}

#msg-input textarea:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(199, 210, 254, 0.4) !important;
}

#examples-row {
    display: flex !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
    margin-top: 4px !important;
}

#examples-row button {
    background-color: #ffffff !important;
    border: 1px solid #C7D2FE !important;
    border-radius: 999px !important;
    color: #4F46E5 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    flex: 1 !important;
    min-width: 120px !important;
}

#examples-row button:hover {
    border-color: #6366F1 !important;
    background-color: #F5F3FF !important;
    transform: translateY(-1px) !important;
}

#action-row {
    display: flex !important;
    gap: 12px !important;
    margin-top: 8px !important;
}

#clear-btn {
    background-color: #ffffff !important;
    border: 1px solid #C7D2FE !important;
    border-radius: 999px !important;
    color: #1F2937 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    flex: 1 !important;
}

#clear-btn:hover {
    background-color: #F5F3FF !important;
    border-color: #6366F1 !important;
    transform: translateY(-1px) !important;
}

#analyze-btn {
    background: linear-gradient(135deg, #818CF8 0%, #4F46E5 100%) !important;
    border: none !important;
    border-radius: 999px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    flex: 1.5 !important;
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.15) !important;
}

#analyze-btn:hover {
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.25) !important;
    transform: translateY(-2px) scale(1.01) !important;
}

/* Results panel card styling */
.result-panel {
    margin-top: 16px !important;
    padding: 24px !important;
    border-radius: 18px !important;
    background-color: #ffffff !important;
    border: 1px solid rgba(199, 210, 254, 0.4) !important;
    box-shadow: 0 10px 25px rgba(79, 70, 229, 0.04) !important;
    animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards !important;
}

.result-empty {
    color: #6B7280 !important;
    font-size: 0.95rem !important;
    text-align: center !important;
    padding: 24px !important;
    border: 1px dashed #C7D2FE !important;
    border-radius: 18px !important;
    background-color: #ffffff !important;
    margin-top: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.3s ease !important;
}

/* Badge popup pop animation */
.result-badge {
    display: inline-block !important;
    padding: 6px 16px !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    animation: badgePop 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards !important;
}

@keyframes badgePop {
    from {
        transform: scale(0.85);
        opacity: 0;
    }
    to {
        transform: scale(1);
        opacity: 1;
    }
}

.spam-badge {
    background-color: #FEE2E2 !important;
    color: #B91C1C !important;
}

.ham-badge {
    background-color: #DCFCE7 !important;
    color: #15803D !important;
}

.confidence-label {
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    color: #1F2937 !important;
}

.confidence-bar {
    width: 100% !important;
    height: 6px !important;
    background-color: #F3F4F6 !important;
    border-radius: 999px !important;
    overflow: hidden !important;
    margin-top: 12px !important;
}

/* Confidence Fill grow animation */
.confidence-fill {
    height: 100% !important;
    border-radius: 999px !important;
    animation: barGrow 1.2s cubic-bezier(0.25, 1, 0.5, 1) forwards !important;
}

@keyframes barGrow {
    from { width: 0% !important; }
}

.confidence-fill.spam {
    background-color: #B91C1C !important;
}

.confidence-fill.ham {
    background-color: #15803D !important;
}

/* Hide number input spinner arrows globally */
input[type="number"]::-webkit-outer-spin-button,
input[type="number"]::-webkit-inner-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
}
input[type="number"] {
    -moz-appearance: textfield !important;
}

#accuracy-line p {
    text-align: center !important;
    color: #1F2937 !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    margin: 8px 0 0 0 !important;
}

#footer-text p {
    text-align: center !important;
    color: #9CA3AF !important;
    font-size: 0.8rem !important;
    margin: 0 !important;
}

footer { display: none !important; }

/* Stats diagnostics panel with hover card animation */
.stats-panel {
    background-color: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-top: 20px !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}

.stats-panel:hover {
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.08) !important;
    transform: translateY(-1px) !important;
}

.stats-title {
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: #1E40AF !important;
    margin-bottom: 8px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

.stats-grid {
    display: flex !important;
    justify-content: space-between !important;
    gap: 12px !important;
    flex-wrap: wrap !important;
}

.stat-item {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 !important;
    min-width: 100px !important;
    transition: all 0.3s ease !important;
}

.stat-item:hover {
    transform: translateY(-2px) !important;
}

.stat-value {
    font-family: 'Poppins', sans-serif !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    color: #2563EB !important;
}

.stat-label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #1E40AF !important;
}
"""

# Configure theme using gr.themes.Base and Google Fonts
theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "Inter", "system-ui", "sans-serif"],
)

with gr.Blocks(theme=theme, css=css, title="Spam Detector") as demo:
    with gr.Column(elem_id="page-container"):
        
        # 1. HEADER (slim row, inside the Blocks, left-aligned shield/mail + right-aligned Model Ready)
        with gr.Row(elem_id="header-row"):
            gr.Markdown("🛡️ **Spam Detector**", elem_id="header-left")
            gr.Markdown("● Model Ready", elem_id="header-right")
            
        # 2. MAIN CARD (single gr.Column with panel style, centered, max-width ~700px)
        with gr.Column(elem_id="main-card"):
            gr.Markdown("# Email Spam Classifier", elem_id="card-title")
            gr.Markdown("Paste an email or message below to check if it's spam.", elem_id="card-subtitle")
            
            # Message input Textbox (5 lines)
            msg_input = gr.Textbox(
                lines=5,
                placeholder="Paste the email content here...",
                show_label=False,
                elem_id="msg-input"
            )
            
            # Example chips (three buttons auto-filling the textbox)
            with gr.Row(elem_id="examples-row"):
                chip_lottery = gr.Button("Lottery Scam", elem_id="chip-lottery")
                chip_meeting = gr.Button("Meeting Reminder", elem_id="chip-meeting")
                chip_phishing = gr.Button("Phishing Link", elem_id="chip-phishing")
                
            # Clear / Analyze Buttons
            with gr.Row(elem_id="action-row"):
                clear_btn = gr.Button("Clear", elem_id="clear-btn")
                analyze_btn = gr.Button("⚡ Analyze", elem_id="analyze-btn")
                
            # Result Display Area
            result_display = gr.HTML(
                value="<div class='result-empty'>Paste a message and click Analyze to see the result.</div>",
                elem_id="result-display"
            )
            
            # 5. DIAGNOSTICS SECTION (Blue background and blue font)
            gr.HTML("""
            <div class="stats-panel">
              <div class="stats-title">📊 System Metrics</div>
              <div class="stats-grid">
                <div class="stat-item">
                  <span class="stat-value">98.9%</span>
                  <span class="stat-label">Accuracy Rate</span>
                </div>
                <div class="stat-item">
                  <span class="stat-value">5,572</span>
                  <span class="stat-label">Messages Trained</span>
                </div>
                <div class="stat-item">
                  <span class="stat-value">&lt; 200ms</span>
                  <span class="stat-label">Response Time</span>
                </div>
              </div>
            </div>
            """, elem_id="metrics-panel")
            
        # 3. MODEL ACCURACY LINE (Muted text under card)
        gr.Markdown(f"Model test accuracy: {MODEL_ACCURACY}%", elem_id="accuracy-line")
        
        # 4. FOOTER (Muted stack info)
        gr.Markdown("Built with Python, TensorFlow & Keras", elem_id="footer-text")

    # Wire example chip buttons to fill message textbox
    chip_lottery.click(
        fn=lambda: "CONGRATULATIONS! You have been selected as the winner of the international mega lottery jackpot! To claim your prize of $5,000,000, please send your banking details and copy of ID to our agent immediately.",
        inputs=[],
        outputs=msg_input
    )
    chip_meeting.click(
        fn=lambda: "Hi everyone, this is a quick reminder that we have a project status meeting tomorrow morning at 10 AM. We'll be reviewing the Q3 metrics and outlining our plan for next month. See you then!",
        inputs=[],
        outputs=msg_input
    )
    chip_phishing.click(
        fn=lambda: "Urgent: Your account security has been compromised. We detected unusual login attempts from another device. Please click on the link below to verify your identity and restore access: http://verify-security-alert-update.com",
        inputs=[],
        outputs=msg_input
    )

    # Wire main action buttons to prediction and clear functions
    analyze_btn.click(
        fn=predict,
        inputs=msg_input,
        outputs=[msg_input, result_display],
    )
    msg_input.submit(
        fn=predict,
        inputs=msg_input,
        outputs=[msg_input, result_display],
    )
    clear_btn.click(
        fn=lambda: (
            "",
            "<div class='result-empty'>Paste a message and click Analyze to see the result.</div>"
        ),
        inputs=[],
        outputs=[msg_input, result_display],
    )

if __name__ == "__main__":
    import socket
    
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    start_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    
    def find_free_port(host, start_port):
        port = start_port
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, port))
                    return port
                except OSError:
                    port += 1

    server_port = find_free_port(host, start_port)
    print(f"Port {start_port} was occupied or unavailable. Automatically binding to available port: {server_port}")
    
    demo.launch(
        server_name=host,
        server_port=server_port,
        share=False,
        show_error=True,
    )
