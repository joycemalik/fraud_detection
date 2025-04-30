

import os
import time
import threading
import wave
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
import pyaudio

# Load environment variables
load_dotenv()
app = Flask(__name__, static_folder="static")
CORS(app)

# Groq client setup
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Audio constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SEC = 5  # seconds per chunk
p = pyaudio.PyAudio()

# Monitoring state & memory
global monitoring, conversation_log, thread
monitoring = False
conversation_log = ""
thread = None
LOG_FILE = "conversation.txt"  # persist conversation

# Fraud scoring parameters
SUSPICIOUS_KEYWORDS = ["bank", "account", "transfer", "otp", "password", "login", "ssn"]
ALPHA = 0.7  # LLM confidence weight
BETA = 0.3   # heuristic weight

# Language support
language_code = "en"
language_map = {"en": "en", "hindi": "hi", "kannada": "kn"}

# Last status globals
global last_transcript, last_fraud, last_score, last_reason
last_transcript = ""
last_fraud = False
last_score = 0.0
last_reason = ""


def record_chunk(duration_secs=RECORD_SEC):
    """Record audio chunk and return bytes."""
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    for _ in range(int(RATE / CHUNK * duration_secs)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    return b"".join(frames)


def save_audio(audio_bytes, filename="temp.wav"):
    """Save audio bytes to WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_bytes)
    return filename


def transcribe(audio_bytes):
    """Transcribe with Groq Whisper in chosen language."""
    path = save_audio(audio_bytes)
    with open(path, "rb") as f:
        resp = groq_client.audio.transcriptions.create(
            file=(os.path.basename(path), f.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            language=language_map.get(language_code, "en")
        )
    return resp.text.strip()


def heuristic_score(text: str) -> float:
    """Compute heuristic keyword frequency score."""
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(words.count(k) for k in SUSPICIOUS_KEYWORDS)
    return min(1.0, count / len(words))


def detect_fraud(context: str):
    """Combine LLM and heuristic to detect fraud."""
    prompt = (
        "You are an AI security agent specialized in fraud detection in the Indian context. "
        "Analyze the entire conversation below and respond ONLY with JSON: {\"fraud\":bool, \"confidence\":float, \"reason\":\"...\"}."
        f"\nConversation:\n'''{context}'''"
    )
    resp = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a strict JSON-only fraud classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=200
    )
    out = resp.choices[0].message.content.strip()
    try:
        obj = json.loads(out)
        llm_conf = obj.get("confidence", 0.0)
        h_score = heuristic_score(context)
        combined = ALPHA * llm_conf + BETA * h_score
        is_fraud = combined > 0.5
        return is_fraud, combined, obj.get("reason", "No reason provided.")
    except json.JSONDecodeError:
        return False, 0.0, "Error parsing LLM response."


def append_log(text: str):
    """Persist transcript."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def monitor_audio():
    """Background thread to record, transcribe, detect, and update status."""
    global conversation_log, last_transcript, last_fraud, last_score, last_reason
    while monitoring:
        audio = record_chunk()
        text = transcribe(audio)
        if text:
            conversation_log += " " + text
            last_transcript = text
            append_log(text)
            fraud, score, reason = detect_fraud(conversation_log)
            last_fraud, last_score, last_reason = fraud, score, reason
            print(f"[TRANSCRIPT] {text}\n[FRAUD] {fraud} @ {score:.2f} – {reason}")
    print("Monitoring stopped.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_monitor():
    global monitoring, thread, conversation_log, language_code
    data = request.get_json() or {}
    language_code = data.get('language', 'en')
    if not monitoring:
        monitoring = True
        conversation_log = ""
        open(LOG_FILE, 'w').close()
        thread = threading.Thread(target=monitor_audio, daemon=True)
        thread.start()
        return jsonify({"message": "Monitoring started.", "language": language_code})
    return jsonify({"message": "Already monitoring."})

@app.route('/stop', methods=['POST'])
def stop_monitor():
    global monitoring
    monitoring = False
    return jsonify({"message": "Monitoring stopped.", "conversation": conversation_log.strip()})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "monitoring": monitoring,
        "last_transcript": last_transcript,
        "fraud": last_fraud,
        "score": last_score,
        "reason": last_reason
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# ---------------------------------------
