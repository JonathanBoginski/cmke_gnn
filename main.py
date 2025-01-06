from fastapi import FastAPI
from app.routes.predict import router as predict_router

app = FastAPI(title="GNN Link Prediction Service", version="0.1.0")

app.include_router(predict_router)

@app.get("/status")
def get_status():
    return {"status": "running"}
