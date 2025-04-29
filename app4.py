import os
import time
import threading
import wave
import numpy as np
import pyaudio
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)
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

# Monitoring state
global monitoring, conversation_log, thread
monitoring = False
conversation_log = ""
thread = None
LOG_FILE = "conversation.txt"  # file to persist conversation

# Heuristic parameters for fraud scoring
SUSPICIOUS_KEYWORDS = ["bank", "account", "transfer", "otp", "password", "login", "ssn"]
ALPHA = 0.7  # weight for LLM confidence
BETA = 0.3   # weight for heuristic keyword frequency


def record_chunk(duration_secs=RECORD_SEC):
    """Record audio for a fixed duration and return bytes and duration."""
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    start = time.time()
    for _ in range(int(RATE / CHUNK * duration_secs)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    end = time.time()
    return b"".join(frames), end - start


def save_audio(audio_bytes, filename="temp.wav"):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_bytes)
    return filename


def transcribe(audio_bytes):
    """Transcribe audio using Groq Whisper."""
    path = save_audio(audio_bytes)
    with open(path, "rb") as f:
        resp = groq_client.audio.transcriptions.create(
            file=(os.path.basename(path), f.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )
    return resp.text.strip()


def heuristic_score(text: str) -> float:
    """Compute heuristic score based on suspicious keyword frequency."""
    words = text.lower().split()
    if not words:
        return 0.0
    count = sum(words.count(k) for k in SUSPICIOUS_KEYWORDS)
    return min(1.0, count / len(words))  # cap at 1.0


def detect_fraud(context: str):
    """Use LLaMA to detect fraud with context and combine with heuristic."""
    # Prepare prompt with full conversation context
    prompt = (
        "You are an AI security agent specialized in fraud detection. "
        "Analyze the entire conversation transcript below for suspicious or fraudulent intent.\n"
        "Respond only with JSON in this format:\n"
        "{\n  \"fraud\": boolean,\n  \"confidence\": float (0.0-1.0),\n  \"reason\": \"Short explanation\"\n}\n\n"
        f"Conversation Context:\n'''{context}'''"
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
        # compute heuristic
        h_score = heuristic_score(context)
        # combined score
        combined = ALPHA * llm_conf + BETA * h_score
        # decide fraud based on combined threshold
        is_fraud = combined > 0.5
        reason = obj.get("reason", "No reason provided.")
        return is_fraud, combined, reason
    except Exception as e:
        print(f"Error parsing JSON from LLaMA: {e}")
        return False, 0.0, "Error in fraud detection."


def append_log(text: str):
    """Persist conversation to file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def monitor_audio():
    global monitoring, conversation_log
    while monitoring:
        audio, dur = record_chunk()
        chunk_txt = transcribe(audio)
        if chunk_txt:
            # update context
            conversation_log += " " + chunk_txt
            append_log(chunk_txt)
            print(f"Chunk transcript: {chunk_txt}")
            # detect fraud with full context
            fraud, score, reason = detect_fraud(conversation_log)
            print(f"-> Fraud: {fraud}, Score: {score:.2f}, Reason: {reason}")
            if fraud:
                print("⚠️ Suspicious conversation detected!")
    print("Monitoring thread exiting.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_monitoring():
    global monitoring, thread, conversation_log
    if not monitoring:
        monitoring = True
        conversation_log = ""
        # clear log file
        open(LOG_FILE, 'w').close()
        thread = threading.Thread(target=monitor_audio)
        thread.start()
        return jsonify({"message": "Monitoring started."})
    return jsonify({"message": "Already running."})

@app.route('/stop', methods=['POST'])
def stop_monitoring():
    global monitoring
    monitoring = False
    # wait for thread to finish
    if thread:
        thread.join()
    return jsonify({"message": "Monitoring stopped.", "conversation": conversation_log.strip()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
