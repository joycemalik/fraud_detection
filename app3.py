# app.py
import os
import threading
import queue
import logging
import time
from typing import Tuple
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import groq
from groq import Groq, GroqError
import pyaudio
import json
from tenacity import retry, stop_after_delay, wait_exponential, retry_if_exception_type
import requests
import numpy as np

# —— Configuration & Logging ——
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_KEY:
    raise RuntimeError("Missing GROQ_API_KEY environment variable")

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logging.info(f"Groq SDK version: {groq.__version__}")  # ([github.com](https://github.com/groq/groq-python))

# —— Connectivity check ——
def check_groq_connectivity(api_key: str):
    client = Groq(api_key=api_key)
    try:
        client.audio.transcriptions.create(
            file=("ping.wav", b""),
            model="whisper-large-v3",
            response_format="verbose_json"
        )
        logging.info("Groq API connectivity check passed.")
    except GroqError as e:
        err = getattr(e, 'args', [None])[-1]
        if isinstance(err, dict) and err.get('error', {}).get('type') == 'invalid_request_error':
            logging.info("Groq API reachable (invalid_request_error acknowledged).")
        else:
            logging.critical(f"Cannot reach Groq API: {e}")
            raise SystemExit("Exiting: no connectivity to Groq Whisper.")
    except requests.exceptions.RequestException as e:
        logging.critical(f"Network error contacting Groq API: {e}")
        raise SystemExit("Exiting: network error to Groq Whisper.")

check_groq_connectivity(GROQ_KEY)

# —— Transcription retry wrapper ——
@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
    stop=stop_after_delay(30),
    reraise=True
)
def safe_transcribe_bytes(client: Groq, audio_bytes: bytes) -> str:
    resp = client.audio.transcriptions.create(
        file=("audio.wav", audio_bytes),
        model="whisper-large-v3",
        response_format="verbose_json"
    )
    return resp.text.strip()

# —— Fraud detection ——
def detect_fraud(client: Groq, text: str) -> Tuple[bool, float]:
    prompt = (
        "Classify whether the following transcript contains fraud or malicious intent.\n"
        "Respond only with JSON: {\"fraud\":true/false,\"confidence\":<0.0-1.0>}\n\n"
        f"Transcript: \"\"\"{text}\"\"\""
    )
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        data = json.loads(response.choices[0].message.content.strip())
        return data.get("fraud", False), data.get("confidence", 0.0)
    except (GroqError, json.JSONDecodeError) as e:
        logging.error(f"Fraud detection error: {e}")
        return False, 0.0

# —— Beep alert ——
def alert_beep():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, output=True)
    t = np.linspace(0, 0.5, int(16000 * 0.5), False)
    tone = np.sin(1000 * 2 * np.pi * t).astype(np.float32)
    stream.write(tone.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

# —— Audio Monitor Class with VAD & Fraud logic ——
class AudioMonitor:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.queue: queue.Queue[bytes] = queue.Queue()
        self.recording = False
        self.log_context = []  # list of events
        self.lock = threading.Lock()

    def start_stream(self):
        if self.stream is None:
            self.stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )

    def record_loop(self):
        self.start_stream()
        silence_threshold = 500
        silence_frames = 0
        buffer = bytearray()

        while self.recording:
            data = self.stream.read(1024, exception_on_overflow=False)
            buffer.extend(data)
            level = max(buffer[-1024:])
            if level < silence_threshold:
                silence_frames += 1
            else:
                silence_frames = 0
            if len(buffer) > 1024*5 and silence_frames > 20:
                self.queue.put(bytes(buffer))
                buffer.clear()
                silence_frames = 0
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def worker_loop(self):
        while self.recording or not self.queue.empty():
            try:
                chunk = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            transcript = safe_transcribe_bytes(self.client, chunk)
            if not transcript:
                continue
            fraud, confidence = detect_fraud(self.client, transcript)
            event = {
                "transcript": transcript,
                "fraud": fraud,
                "confidence": confidence
            }
            with self.lock:
                self.log_context.append(event)
            logging.info(f"Transcript: {transcript}")  # ([console.groq.com](https://console.groq.com/docs/speech-to-text))
            logging.info(f"Fraud={fraud}, Confidence={confidence:.2f}")
            if fraud and confidence > 0.7:
                logging.warning(f"⚠️ Fraud detected! Confidence={confidence:.2f}")
                alert_beep()

    def start(self):
        self.recording = True
        threading.Thread(target=self.record_loop, daemon=True).start()
        threading.Thread(target=self.worker_loop, daemon=True).start()

    def stop(self):
        self.recording = False

monitor = AudioMonitor(GROQ_KEY)

# —— Flask app ——
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/start", methods=["POST"])
def start_monitoring():
    if not monitor.recording:
        monitor.log_context = []
        monitor.start()
        return jsonify({"message": "Monitoring started."}), 200
    return jsonify({"message": "Already running."}), 400

@app.route("/stop", methods=["POST"])
def stop_monitoring():
    monitor.stop()
    return jsonify({"message": "Monitoring stopped.", "events": monitor.log_context}), 200

@app.route("/status", methods=["GET"])
def status():
    with monitor.lock:
        return jsonify({"events": monitor.log_context}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
