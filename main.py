from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content
import json, os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- CONFIGURATION FIX ---
# Default to your specific project ID to avoid the 'audio-safety-agent' mismatch
PROJECT = os.getenv("GCP_PROJECT", "project-34774df1-016e-4017-a26")
vertexai.init(project=PROJECT, location="us-central1")

# Load Fingerprints
try:
    with open("fingerprints.json") as f:
        FP_DATA = json.load(f)
        URGENT_LIST = json.dumps(FP_DATA.get("urgent_sounds", []))
except Exception:
    URGENT_LIST = "[]"

# --- PROMPT DEFINITIONS ---
WORKER_SYSTEM = """You are a sound classifier. 
Output EXACTLY one word only:
CHIME (if you hear a bell, doorbell, or tonal sound)
VOICE (if you hear a clear human voice)
BACKGROUND (music, noise, silence, or anything else)"""

SUPER_SYSTEM = """Role: Home Safety Supervisor AI.
Core Mandate: Evaluate incoming sound classifications against a provided "Urgent List."
Output Constraint: Output EXACTLY one word: URGENT or NORMAL."""

@app.get("/health")
def health(): 
    return {"status": "ok"}

@app.post("/analyze_audio")
async def analyze(audio: UploadFile = File(...)):
    data = await audio.read()
    audio_part = Part.from_data(data, mime_type="audio/wav")
    
    # 1. The Worker: Use system_instruction for behavior
    worker_model = GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=WORKER_SYSTEM
    )
    
    # Simplified call: system_instruction handles the 'rules' 
    worker_response = worker_model.generate_content([audio_part])
    signal = worker_response.text.strip().upper()

    # Early exit for background noise
    if "BACKGROUND" in signal:
        return {
            "verdict": "NORMAL",
            "worker_signal": signal,
            "reasoning": "Background noise is filtered automatically."
        }

    # 2. The Supervisor
    supervisor_model = GenerativeModel(
        "gemini-1.5-pro",
        system_instruction=SUPER_SYSTEM
    )

    super_prompt = f"""Sound detected: {signal}
Known urgent sounds: {URGENT_LIST}

Does the detected sound match any urgent sound?"""

    super_response = supervisor_model.generate_content(super_prompt)
    verdict_raw = super_response.text.strip().upper()

    return {
        "verdict": "URGENT" if "URGENT" in verdict_raw else "NORMAL",
        "worker_signal": signal,
        "reasoning": verdict_raw
    }

if __name__ == "__main__":
    import uvicorn
    # Use the $PORT environment variable for Cloud Run compatibility
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
