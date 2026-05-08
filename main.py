from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import json, os

app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"])

PROJECT = os.getenv("GCP_PROJECT", "audio-safety-agent")
vertexai.init(project=PROJECT, location="us-central1")

with open("fingerprints.json") as f:
    FP = json.load(f)

WORKER = "Listen to this audio. Output ONLY one word: CHIME, VOICE, or BACKGROUND. No other text."

SUPER = """Sound detected: {signal}
Known urgent sounds: {fp}
Output ONLY: URGENT or NORMAL"""

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/analyze_audio")
async def analyze(audio: UploadFile = File(...)):
    data = await audio.read()
    part = Part.from_data(data, mime_type="audio/wav")
    w = GenerativeModel("gemini-1.5-flash")
    signal = w.generate_content([WORKER, part]).text.strip().upper()
    if "BACKGROUND" in signal:
        return {"verdict": "NORMAL", "worker": signal}
    s = GenerativeModel("gemini-1.5-pro")
    prompt = SUPER.format(signal=signal, fp=json.dumps(FP["urgent_sounds"]))
    verdict = s.generate_content(prompt).text.strip()
    return {"verdict": "URGENT" if "URGENT" in verdict else "NORMAL",
            "worker": signal, "reasoning": verdict}