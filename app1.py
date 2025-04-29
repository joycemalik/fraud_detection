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

load_dotenv()

app = Flask(__name__)
CORS(app)

# Groq setup
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Audio constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SEC = 5  # 5 seconds recording chunks
p = pyaudio.PyAudio()

# Global variables
monitoring = False
conversation_log = ""
thread = None

def record_chunk(duration_secs=5):
    """Record audio chunk."""
    print(f"🎙️ Recording {duration_secs} seconds...")
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    start_time = time.time()
    for _ in range(int(RATE / CHUNK * duration_secs)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    end_time = time.time()
    duration = end_time - start_time
    print("✅ Recording done.")
    return b"".join(frames), duration

def save_audio(audio_bytes, filename="temp.wav"):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_bytes)
    return filename

def transcribe(audio_bytes):
    audio_path = save_audio(audio_bytes)
    with open(audio_path, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=("temp.wav", file.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )
    return transcription.text.strip()

def detect_fraud(text):
    """Use LLaMA model to detect fraud."""
    prompt = (
        "You are an AI security agent specialized in fraud detection.\n"
        "Analyze the provided conversation transcript carefully.\n\n"
        "Task:\n"
        "- If fraud, scam, financial manipulation, or sensitive data theft is suspected, classify as fraud.\n"
        "- Otherwise, classify as safe.\n\n"
        "Output ONLY valid JSON strictly in the following format:\n"
        "{\n"
        "  \"fraud\": true or false,\n"
        "  \"confidence\": float (between 0.0 and 1.0),\n"
        "  \"reason\": \"Short sentence explanation\"\n"
        "}\n\n"
        "Conversation Transcript:\n"
        f"\"\"\"\n{text}\n\"\"\"\n"
    )
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a strict JSON-only fraud classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=200
    )
    out = response.choices[0].message.content.strip()
    try:
        fraud_obj = json.loads(out)
        return fraud_obj.get("fraud", False), fraud_obj.get("confidence", 0.0), fraud_obj.get("reason", "No reason provided.")
    except Exception as e:
        print(f"⚠️ JSON parsing error: {e}")
        return False, 0.0, "Error in processing."

def monitor_audio():
    """Background thread to monitor audio continuously."""
    global monitoring, conversation_log
    while monitoring:
        try:
            audio_bytes, duration = record_chunk(RECORD_SEC)
            transcript = transcribe(audio_bytes)
            if transcript:
                conversation_log += " " + transcript
                print(f"📝 New transcript chunk: {transcript}")

                # Analyze continuously
                is_fraud, conf, reason = detect_fraud(transcript)
                print(f"🔍 Fraud Detected?: {is_fraud} | \nConfidence: {conf:.2f} |\n Reason: {reason}")

                if is_fraud and conf > 0.9:
                    print("⚠️ WARNING: Suspicious conversation detected!")

        except Exception as e:
            print(f"⚠️ Error while monitoring: {e}")
            break

    print("🛑 Monitoring Stopped.")

# Frontend Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_monitoring():
    global monitoring, thread, conversation_log
    if not monitoring:
        monitoring = True
        conversation_log = ""
        thread = threading.Thread(target=monitor_audio)
        thread.start()
        return jsonify({"message": "Started monitoring audio."})
    else:
        return jsonify({"message": "Already monitoring."})

@app.route('/stop', methods=['POST'])
def stop_monitoring():
    global monitoring, conversation_log
    monitoring = False
    time.sleep(1)  # Give time to safely stop
    return jsonify({
        "message": "Stopped monitoring audio.",
        "conversation": conversation_log.strip()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
