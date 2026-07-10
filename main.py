from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is live"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    return {"risk": 0.5}
