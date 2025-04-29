import os, time, threading, wave, json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
import pyaudio

# ── Config & client setup ─────────────────────────────
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)
CORS(app)

# ── Audio settings ────────────────────────────────────
CHUNK, FORMAT, CHANNELS, RATE = 1024, pyaudio.paInt16, 1, 16000
RECORD_SEC = 5
p = pyaudio.PyAudio()

# ── Global state ──────────────────────────────────────
monitoring = False
conversation_log = ""
thread = None
LOG_FILE = "conversation.txt"

# Live-status globals
latest_score = 0.0
latest_is_fraud = False

# ── Fraud logic params ─────────────────────────────────
DECISION_THRESHOLD = 0.5  # LLM-only threshold
# “Must-flag” phrases: any match → guaranteed fraud
SUSPICIOUS_PHRASES = [
    "share otp", "processing fee", "lottery", "winner", "kbc",
    "pm kisan", "account blocked", "urgent action", "loan disbursal"
]

def record_chunk():
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    for _ in range(int(RATE/CHUNK * RECORD_SEC)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    return b"".join(frames)

def save_audio(audio_bytes, fn="temp.wav"):
    with wave.open(fn, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_bytes)
    return fn

def transcribe(audio_bytes, language="en"):
    fn = save_audio(audio_bytes)
    with open(fn, "rb") as f:
        resp = groq_client.audio.transcriptions.create(
            file=(os.path.basename(fn), f.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            language=language
        )
    return resp.text.strip()

def detect_fraud(text, llm_conf):
    """Combine LLM confidence + phrase-flag logic."""
    txt = text.lower()
    # 1) phrase flag
    flag = any(ph in txt for ph in SUSPICIOUS_PHRASES)
    if flag:
        return True, 1.0  # guaranteed fraud

    # 2) otherwise use LLM’s confidence
    is_fraud = llm_conf >= DECISION_THRESHOLD
    return is_fraud, llm_conf

def get_llm_confidence(context):
    """Ask LLM for a fraud confidence (0.0–1.0)."""
    prompt = (
        "You are an AI agent specialized in Indian phone-call fraud detection.\n"
        "Respond only with JSON: { \"fraud\": boolean, \"confidence\": float, \"reason\": string }\n\n"
        f"Conversation:\n'''{context}'''"
    )
    resp = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role":"system","content":"JSON-only fraud classifier."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0,
        max_tokens=100
    )
    try:
        obj = json.loads(resp.choices[0].message.content.strip())
        return float(obj.get("confidence", 0.0)), obj.get("reason","")
    except:
        return 0.0, "LLM parse error"

def monitor_audio(lang):
    global monitoring, conversation_log, latest_score, latest_is_fraud
    while monitoring:
        audio = record_chunk()
        chunk_txt = transcribe(audio, language=lang)
        if chunk_txt:
            conversation_log += " " + chunk_txt
            with open(LOG_FILE,'a',encoding='utf-8') as f:
                f.write(chunk_txt + "\n")

            # 1) get LLM confidence
            llm_conf, reason = get_llm_confidence(conversation_log)

            # 2) combine with phrase logic
            is_fraud, combined = detect_fraud(conversation_log, llm_conf)

            # 3) update live globals
            latest_score    = combined
            latest_is_fraud = is_fraud

            print(f"[{lang}] “{chunk_txt}” → Fraud={is_fraud}, Score={combined:.2f}, Reason={reason}")

    print("🛑 Monitoring stopped.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_monitoring():
    global monitoring, thread, conversation_log, latest_score, latest_is_fraud
    if not monitoring:
        monitoring = True
        conversation_log = ""
        latest_score = 0.0
        latest_is_fraud = False
        open(LOG_FILE,'w').close()

        lang = request.json.get("language","en")
        thread = threading.Thread(target=monitor_audio, args=(lang,))
        thread.start()
        return jsonify({"message":"started","language":lang})
    return jsonify({"message":"already running"})

@app.route('/stop', methods=['POST'])
def stop_monitoring():
    global monitoring
    monitoring = False
    if thread:
        thread.join()
    return jsonify({"message":"stopped","conversation":conversation_log.strip()})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "fraud": latest_is_fraud,
        "score": latest_score
    })

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)
