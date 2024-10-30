from fastapi import FastAPI
import pandas as pd
import pickle

# Load the trained model and scaler
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI on Vercel!"}

@app.post("/predict")
async def predict_rate(entity_id: int, job_title_encoded: int, certifications: list):
    results = []
    for cert in certifications:
        data = pd.DataFrame({'entity_id': [entity_id], 'certificate': [cert], 'lang_id': [job_title_encoded]})
        data_scaled = scaler.transform(data)
        pred = best_model.predict(data_scaled)
        results.append({"certification": cert, "predicted_rate": pred[0]})
    return {"predictions": results}
