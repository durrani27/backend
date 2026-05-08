from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import vertexai
from vertexai.generative_models import (
    GenerativeModel, Part, Content  # same as yours
)
import json, os

app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"])

PROJECT = os.getenv("GCP_PROJECT", "project-34774df1-016e-4017-a26")
vertexai.init(project=PROJECT, location="us-central1")

try:
    with open("fingerprints.json") as f:
        FP_DATA = json.load(f)
        URGENT_LIST = json.dumps(FP_DATA.get("urgent_sounds", []))
except Exception:
    URGENT_LIST = "[]"

# --- WORKER SYSTEM PROMPT (same as yours, matches Studio) ---
WORKER_SYSTEM = """You are a sound classifier.
Output EXACTLY one word only:
CHIME (if you hear a bell, doorbell, or tonal sound)
VOICE (if you hear a clear human voice)
BACKGROUND (music, noise, silence, or anything else)"""

# --- SUPERVISOR SYSTEM PROMPT (CHANGED: added Decision Logic section) ---
SUPER_SYSTEM = """Role: Home Safety Supervisor AI.
Core Mandate: Evaluate incoming sound classifications against a provided Urgent List.
Privacy Protocol: Do NOT transcribe or repeat any speech or audio details.
Output Constraint: Output EXACTLY one word: URGENT or NORMAL. No punctuation. No explanation.

Decision Logic:
- Compare the detected sound type (CHIME, VOICE, BACKGROUND) to the Urgent List.
- If the detected sound type matches any entry in the Urgent List: output URGENT.
- If the detected sound is BACKGROUND, or does not match: output NORMAL."""

# --- FEW-SHOT EXAMPLES FOR WORKER (ADDED from Studio generated code) ---
# These teach the Worker by showing it correct examples before your real audio
WORKER_EXAMPLES = [
    Content(role="user", parts=[
        Part.from_text("I heard a high-pitched two-tone ding-dong chime lasting 1 second.")
    ]),
    Content(role="model", parts=[Part.from_text("CHIME")]),
    Content(role="user", parts=[Part.from_text("I heard YouTube music playing.")]),
    Content(role="model", parts=[Part.from_text("BACKGROUND")]),
    Content(role="user", parts=[Part.from_text("An adult woman called a name loudly twice.")]),
    Content(role="model", parts=[Part.from_text("VOICE")]),
]

# --- FEW-SHOT EXAMPLES FOR SUPERVISOR (ADDED from Studio generated code) ---
SUPER_EXAMPLES = [
    Content(role="user", parts=[Part.from_text(
        'Sound detected: CHIME\nKnown urgent sounds: ["Two-tone doorbell"]\nDoes it match?'
    )]),
    Content(role="model", parts=[Part.from_text("URGENT")]),
    Content(role="user", parts=[Part.from_text(
        'Sound detected: BACKGROUND\nKnown urgent sounds: ["Two-tone doorbell"]\nDoes it match?'
    )]),
    Content(role="model", parts=[Part.from_text("NORMAL")]),
    Content(role="user", parts=[Part.from_text(
        'Sound detected: VOICE\nKnown urgent sounds: ["Smoke detector beep"]\nDoes it match?'
    )]),
    Content(role="model", parts=[Part.from_text("NORMAL")]),
]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze_audio")
async def analyze(audio: UploadFile = File(...)):
    data = await audio.read()
    audio_part = Part.from_data(data, mime_type="audio/wav")

    # 1. WORKER — CHANGED: added few-shot examples before audio
    worker_model = GenerativeModel(
        "gemini-1.5-flash",          # keep your model, NOT gemini-3.1-flash-lite
        system_instruction=WORKER_SYSTEM
    )
    worker_response = worker_model.generate_content(
        WORKER_EXAMPLES + [          # ADDED: examples first, then real audio
            Content(role="user", parts=[audio_part])
        ]
    )
    signal = worker_response.text.strip().upper()

    if "BACKGROUND" in signal:
        return {"verdict": "NORMAL", "worker_signal": signal,
                "reasoning": "Background noise filtered."}

    # 2. SUPERVISOR — CHANGED: added few-shot examples + better prompt
    supervisor_model = GenerativeModel(
        "gemini-1.5-pro",            # keep your model, NOT gemini-2.5-pro
        system_instruction=SUPER_SYSTEM   # CHANGED: now uses longer prompt
    )
    super_prompt = f"""Sound detected: {signal}
Known urgent sounds: {URGENT_LIST}
Does the detected sound match any urgent sound?"""

    super_response = supervisor_model.generate_content(
        SUPER_EXAMPLES + [           # ADDED: examples first, then real query
            Content(role="user", parts=[Part.from_text(super_prompt)])
        ]
    )
    verdict_raw = super_response.text.strip().upper()

    return {
        "verdict": "URGENT" if "URGENT" in verdict_raw else "NORMAL",
        "worker_signal": signal,
        "reasoning": verdict_raw
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
