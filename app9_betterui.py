# app.py
import os
import time
import threading
import wave
import pyaudio
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

# —— CONFIG ——
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# —— GLOBAL STATE ——
monitoring = False
conversation_log = ""
worker_thread = None

last_transcript = ""
last_fraud = False
last_confidence = 0.0
last_reason = ""

# —— AUDIO SETTINGS ——
CHUNK       = 1024
FORMAT      = pyaudio.paInt16
CHANNELS    = 1
RATE        = 16000
RECORD_SEC  = 3      
p           = pyaudio.PyAudio()

# —— KEYWORDS (passed into LLM prompt only) ——
SUSPICIOUS_KEYWORDS = ["bank","account","transfer","otp","password","login","ssn"]

def record_chunk(duration=RECORD_SEC):
    """Record `duration` seconds and return raw bytes."""
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    frames = []
    for _ in range(int(RATE/CHUNK * duration)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
    stream.stop_stream(); stream.close()
    return b"".join(frames)

def save_audio(data, fn="temp.wav"):
    """Write WAV to disk."""
    with wave.open(fn, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(data)
    return fn

def transcribe(audio_bytes, lang="en"):
    """Whisper STT via Groq."""
    path = save_audio(audio_bytes)
    with open(path, "rb") as f:
        resp = groq_client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            language=lang
        )
    return resp.text.strip()

def detect_fraud(context, keywords):
    """LLM-only fraud check; returns (bool, confidence, reason)."""
    prompt = (
        "You are an AI fraud detector.  Use these keywords as cues: "
        + ", ".join(keywords) + ".\n"
        "Analyze the conversation below and RESPOND ONLY WITH JSON in this format:\n"
        "{ \"fraud\": boolean, \"confidence\": float, \"reason\": \"...\" }\n\n"
        f"Conversation:\n'''{context}'''"
    )
    resp = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role":"system","content":"Strict JSON-only fraud classifier."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0,
        max_tokens=150
    )
    data = json.loads(resp.choices[0].message.content.strip())
    return (
        data.get("fraud", False),
        float(data.get("confidence", 0.0)),
        data.get("reason","")
    )

def append_log(text):
    with open("conversation.txt","a",encoding="utf-8") as f:
        f.write(text + "\n")

def monitor(lang="en"):
    """Background loop: record→transcribe→classify→update globals."""
    global monitoring, conversation_log
    global last_transcript, last_fraud, last_confidence, last_reason

    while monitoring:
        audio = record_chunk()
        text  = transcribe(audio, lang)
        if text:
            conversation_log += " " + text
            append_log(text)
            last_transcript = text
            fraud, conf, reason = detect_fraud(conversation_log, SUSPICIOUS_KEYWORDS)
            last_fraud       = fraud
            last_confidence  = conf
            last_reason      = reason
            print(f"[{time.strftime('%H:%M:%S')}] «{text}» → fraud={fraud}, conf={conf:.2f}")
    print(">> Monitor stopped.")

# —— FLASK APP ——
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")  # keep your existing UI here

@app.route("/start", methods=["POST"])
def start_monitoring():
    global monitoring, worker_thread, conversation_log
    if not monitoring:
        monitoring = True
        conversation_log = ""
        open("conversation.txt","w").close()
        lang = request.json.get("language", "en")
        worker_thread = threading.Thread(target=monitor, args=(lang,), daemon=True)
        worker_thread.start()
        return jsonify({"message":"started"})
    return jsonify({"message":"already running"}), 400

@app.route("/stop", methods=["POST"])
def stop_monitoring():
    global monitoring
    monitoring = False
    if worker_thread:
        worker_thread.join()
    return jsonify({
        "message":"stopped",
        "conversation": conversation_log.strip()
    })

@app.route("/status")
def status():
    """Front-end can poll this every 500 ms for updates."""
    return jsonify({
        "monitoring":      monitoring,
        "last_transcript": last_transcript,
        "fraud":           last_fraud,
        "confidence":      last_confidence,
        "reason":          last_reason
    })

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
