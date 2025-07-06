from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np, pickle

class IrisFeatures(BaseModel):
    features: list

with open("app/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Iris Classifier using FastAPI"}

@app.post("/predict")
def predict(data: IrisFeatures):
    try:
        x = np.array(data.features).reshape(1, -1)
        y = model.predict(x)
        return {"prediction": int(y[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
