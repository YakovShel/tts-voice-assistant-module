from fastapi import FastAPI

app = FastAPI(title="TTS Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/synthesize")
def synthesize():
    return {"message": "synthesis will be here"}