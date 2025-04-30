import os
import time
import threading
import wave
import json
import io           # <-- Added for in-memory WAV
import atexit       # <-- Added for cleanup
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
import pyaudio
import beepy        # <-- Added for beep sound (install with: pip install beepy)

# ── Config & client setup ─────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")
groq_client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)
CORS(app)

# ── Audio settings ────────────────────────────────────
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SEC = 3  # Reduced chunk duration for potentially faster feedback loop
p = pyaudio.PyAudio()

# Register cleanup function
atexit.register(p.terminate)

# ── Global state ──────────────────────────────────────
monitoring = False
monitoring_lock = threading.Lock() # Lock for thread-safe access to monitoring flag
conversation_log = ""
monitor_thread = None
LOG_FILE = "conversation.txt"

# Live-status globals
latest_score = 0.0
latest_is_fraud = False
status_message = "Idle" # Provides context on monitoring status

# ── Fraud logic params ─────────────────────────────────
# LLM confidence threshold (0–1) used *within* detect_fraud
DECISION_THRESHOLD = 0.5
# Final score threshold (0–1) to trigger fraud alert and stop monitoring
FRAUD_SCORE_THRESHOLD = 0.85 # <--- IMPORTANT: Threshold to trigger the stop

# “Must-flag” phrases: any match → guaranteed fraud
SUSPICIOUS_PHRASES = [
    "share otp", "processing fee", "lottery", "winner", "kbc",
    "pm kisan", "account blocked", "urgent action", "loan disbursal",
    "urgent transfer", "wire immediately", "confirm your password",
    "gift card", "google play card", "apple gift card", "bank details",
    "verify your account", "tax refund", "customs duty"
]

# --- Helper Functions ---

def format_audio_as_wav_bytes(audio_frames):
    """Formats raw audio frames into WAV format in memory."""
    try:
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(audio_frames)
            return wav_buffer.getvalue()
    except Exception as e:
        print(f"Error formatting WAV bytes: {e}")
        return None

def record_chunk():
    """Records a chunk of audio from the default input device."""
    stream = None
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)
        print(f"[*] Recording {RECORD_SEC} seconds...")
        frames = []
        for _ in range(int(RATE / CHUNK * RECORD_SEC)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except IOError as e:
                # Handle buffer overflow or input device issues gracefully
                print(f"Audio read error: {e}")
                # Optionally decide whether to continue or stop
                return None # Indicate failure
        print("[*] Recording finished.")
        return b"".join(frames)
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        return None
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"Error closing audio stream: {e}")


def transcribe(audio_bytes, language="en"):
    """Transcribes audio bytes using Groq API."""
    if not audio_bytes:
        print("Transcription skipped: No audio data.")
        return ""

    wav_data = format_audio_as_wav_bytes(audio_bytes)
    if not wav_data:
        print("Transcription skipped: Failed to format WAV.")
        return ""

    try:
        with io.BytesIO(wav_data) as audio_file:
            # Pass the in-memory bytes directly
            resp = groq_client.audio.transcriptions.create(
                file=("audio.wav", audio_file), # API needs a filename hint
                model="whisper-large-v3", # Using standard whisper-large-v3, adjust if turbo needed/available
                response_format="verbose_json",
                language=language
            )
        # print(f"Transcription API Response: {resp}") # Debugging
        return resp.text.strip()
    except Exception as e:
        print(f"Error during transcription API call: {e}")
        return ""

def detect_fraud(text: str, llm_conf: float):
    """
    Combines keyword matching and LLM confidence for fraud detection.
    Returns (is_fraud: bool, score: float[0–1]):
    - If any suspicious phrase appears → (True, 1.0)
    - Else if llm_conf >= DECISION_THRESHOLD → boost score (capped at 1.0)
    - Else → use dampened score
    """
    txt_lower = text.lower()

    # 1) Immediate flag based on keywords
    for ph in SUSPICIOUS_PHRASES:
        # Use word boundaries for more precise matching (optional but recommended)
        # import re
        # if re.search(r'\b' + re.escape(ph) + r'\b', txt_lower):
        if ph in txt_lower: # Simpler check for now
            print(f"FLAGGED by suspicious phrase: '{ph}'")
            return True, 1.0 # High confidence if keyword matches

    # 2) Otherwise, adjust LLM confidence
    # Ensure llm_conf is within [0, 1]
    llm_conf = max(0.0, min(1.0, llm_conf))

    if llm_conf >= DECISION_THRESHOLD:
        # Boost confidence, but don't exceed 1.0
        # Using simple boost, adjust logic if needed (e.g., diminishing returns)
        boosted_score = min(1.0, llm_conf * 1.2) # Slightly stronger boost
        return True, boosted_score # Considered fraud if LLM is confident enough
    else:
        # Use the (potentially dampened) LLM score directly if below threshold
        # dampening can sometimes hide rising suspicion, consider using llm_conf directly
        # damped = llm_conf * 0.9
        return False, llm_conf # Not considered fraud based on this logic, score reflects LLM belief

def get_llm_confidence(context: str):
    """Ask LLM for a fraud confidence (0.0–1.0) and reason."""
    if not context.strip():
        return 0.0, "No conversation yet."

    prompt = (
        "You are an AI agent specialized in detecting potential fraud in phone call transcripts, particularly common scams in India.\n"
        "Analyze the following conversation snippet and determine the likelihood of it being a fraudulent call.\n"
        "Focus on identifying scam tactics like urgency, requests for sensitive information (OTP, passwords, bank details), fake lotteries, impersonation, processing fees, etc.\n"
        "Respond ONLY with a valid JSON object containing these keys:\n"
        "  \"fraud\": boolean (true if you suspect fraud, false otherwise)\n"
        "  \"confidence\": float (your confidence level from 0.0 to 1.0 for the 'fraud' assessment)\n"
        "  \"reason\": string (a brief explanation for your assessment)\n\n"
        f"Conversation Transcript:\n'''{context}'''\n\n"
        "JSON Response:"
    )
    try:
        resp = groq_client.chat.completions.create(
            model="llama3-70b-8192", # Or consider llama3-8b-8192 for potentially faster/cheaper analysis if acceptable
            messages=[
                {"role": "system", "content": "You are a JSON-only fraud analysis assistant for phone calls."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Lower temperature for more deterministic analysis
            max_tokens=150,  # Increased slightly for potentially longer reasons
            response_format={"type": "json_object"}, # Enforce JSON output if model supports it
        )
        # print(f"LLM API Response: {resp.choices[0].message.content}") # Debugging
        content = resp.choices[0].message.content.strip()
        # Basic cleanup in case of markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        obj = json.loads(content)
        confidence = float(obj.get("confidence", 0.0))
        reason = obj.get("reason", "No reason provided.")
        # Clamp confidence just in case LLM returns value outside [0,1]
        confidence = max(0.0, min(1.0, confidence))
        return confidence, reason
    except json.JSONDecodeError as e:
        print(f"LLM JSON parsing error: {e}. Raw content: '{content}'")
        return 0.0, "LLM response parsing error"
    except Exception as e:
        print(f"Error during LLM API call: {e}")
        return 0.0, f"LLM API error: {e}"

def trigger_fraud_alert():
    """Plays a beep sound and prints an alert message."""
    global status_message
    print("\n" + "="*40)
    print("🚨🚨🚨 FRAUD ALERT DETECTED! 🚨🚨🚨")
    print("="*40 + "\n")
    status_message = "FRAUD DETECTED! Monitoring stopped."
    try:
        beepy.beep(sound='error') # Play an alert sound
    except Exception as e:
        print(f"Could not play beep sound: {e}") # Log if beepy fails

# --- Monitoring Thread ---

def monitor_audio(lang):
    global monitoring, conversation_log, latest_score, latest_is_fraud, status_message
    print(f"🎙️ Starting audio monitoring in language: {lang}")

    while True:
        with monitoring_lock:
            if not monitoring:
                break # Exit loop if monitoring flag is set to False

        audio_chunk = record_chunk()
        if not audio_chunk:
            print("Failed to record audio chunk, continuing...")
            time.sleep(1) # Avoid busy-looping on record errors
            continue

        # --- Transcription ---
        start_time = time.time()
        chunk_txt = transcribe(audio_chunk, language=lang)
        transcription_time = time.time() - start_time
        # print(f"Transcription time: {transcription_time:.2f}s") # Debugging

        if chunk_txt:
            print(f"   [Transcription] ({lang}): \"{chunk_txt}\"")
            # Append to conversation log (thread-safe access not strictly needed here as only this thread writes)
            conversation_log += " " + chunk_txt
            try:
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {chunk_txt}\n")
            except IOError as e:
                print(f"Error writing to log file {LOG_FILE}: {e}")

            # --- Fraud Analysis ---
            start_time = time.time()
            # 1) Get LLM confidence based on the *updated* conversation context
            llm_conf, reason = get_llm_confidence(conversation_log)

            # 2) Combine LLM score with keyword analysis using detect_fraud
            is_potentially_fraud, combined_score = detect_fraud(conversation_log, llm_conf)
            analysis_time = time.time() - start_time
            # print(f"Analysis time: {analysis_time:.2f}s") # Debugging


            # 3) Update global status (accessible by /status route)
            # Consider adding locks if strict guarantees are needed, but for status polling it might be okay
            latest_score = combined_score
            latest_is_fraud = is_potentially_fraud # Reflects the immediate check result

            print(f"   [Analysis] LLM Confidence: {llm_conf:.2f}, Combined Score: {combined_score:.2f}, Fraud Suspected: {is_potentially_fraud}, Reason: {reason}")

            # --- ACTION: Check Threshold ---
            if combined_score >= FRAUD_SCORE_THRESHOLD:
                with monitoring_lock:
                    if monitoring: # Double check if monitoring wasn't stopped externally
                        trigger_fraud_alert()
                        monitoring = False # Stop monitoring
                        # No break here, let the loop check the flag at the top

        else:
            print("   [Transcription] No text detected in chunk.")

        # Small delay to prevent overly tight loop if transcription is very fast or empty
        # time.sleep(0.1)

    print("🛑 Monitoring loop stopped.")
    # Update status message if stopped manually (not by fraud alert)
    with monitoring_lock:
      if status_message != "FRAUD DETECTED! Monitoring stopped.":
          status_message = "Monitoring stopped manually."


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_monitoring():
    """Starts the audio monitoring thread."""
    global monitoring, monitor_thread, conversation_log, latest_score, latest_is_fraud, status_message
    with monitoring_lock:
        if monitoring:
            return jsonify({"message": "Monitoring is already running."}), 400

        print("\n[*] Received request to start monitoring...")
        monitoring = True
        conversation_log = "" # Reset conversation log
        latest_score = 0.0
        latest_is_fraud = False
        status_message = "Monitoring active..."

        # Clear log file content
        try:
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                f.write(f"--- Log Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        except IOError as e:
             print(f"Error clearing log file {LOG_FILE}: {e}")


        lang = request.json.get("language", "en")
        if lang not in ["en", "hi", "es", "fr"]: # Example supported languages, adjust as needed
            print(f"Warning: Unsupported language '{lang}' requested, defaulting to 'en'.")
            lang = "en"

        # Start the monitoring thread
        monitor_thread = threading.Thread(target=monitor_audio, args=(lang,), daemon=True) # Use daemon thread
        monitor_thread.start()

        print(f"[*] Monitoring thread started successfully for language: {lang}")
        return jsonify({"message": "Monitoring started successfully.", "language": lang})

@app.route('/stop', methods=['POST'])
def stop_monitoring():
    """Stops the audio monitoring thread."""
    global monitoring, monitor_thread, status_message
    with monitoring_lock:
        if not monitoring:
            return jsonify({"message": "Monitoring is not currently running."}), 400

        print("\n[*] Received request to stop monitoring...")
        monitoring = False # Signal the thread to stop

    # Wait for the thread to finish gracefully
    if monitor_thread and monitor_thread.is_alive():
        print("[*] Waiting for monitoring thread to complete...")
        monitor_thread.join(timeout=RECORD_SEC + 5) # Wait a bit longer than record time
        if monitor_thread.is_alive():
             print("[Warning] Monitoring thread did not stop gracefully.")
             # Force stop might be needed in some scenarios, but can lead to resource leaks
        else:
             print("[*] Monitoring thread stopped.")

    monitor_thread = None # Clear thread object
    # Avoid overwriting fraud message if stopped due to fraud
    if status_message != "FRAUD DETECTED! Monitoring stopped.":
        status_message = "Monitoring stopped by user request."

    # Return the final conversation log
    final_log = conversation_log.strip()
    return jsonify({"message": status_message, "conversation": final_log})

@app.route('/status', methods=['GET'])
def status():
    """Returns the current fraud detection status."""
    with monitoring_lock:
        monitoring_active = monitoring

    # Return current status, including whether monitoring is active and why it might have stopped
    return jsonify({
        "monitoring_active": monitoring_active,
        "fraud_detected": latest_is_fraud,
        "fraud_score": round(latest_score, 3), # Round score for cleaner display
        "status_message": status_message
    })

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Fraud Detection Server...")
    # Ensure PyAudio terminates properly on exit (redundant with atexit but safe)
    # try:
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=True can cause issues with threading and cleanup
    # finally:
    #     print("Terminating PyAudio...")
    #     p.terminate() # Already handled by atexit