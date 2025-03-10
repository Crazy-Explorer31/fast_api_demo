from contextlib import asynccontextmanager
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import json
import pandas as pd
from save_model import FeaturesGenerator, ColumnDropper

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def price_predictor(x: dict) -> dict:
    with open("model.pkl", 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    x_df = pd.DataFrame(x, index=[0])
    res = loaded_model.predict(x_df)[0]
    return {"prediction": res}

ml_models = {}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    with open("model.pkl", 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    ml_models["price_predictor"] = loaded_model
    yield
    ml_models.clear()

app = FastAPI(lifespan=ml_lifespan_manager)

@app.get("/")
async def root():
    return {
        "Name": "Cars Prediction",
        "description": "This is a cars' price prediction model based on their typical features.",
    }

@app.post("/predict_item_from_json")
async def predict_item_from_json(item: Item):
    print(item)
    return {'result' : ml_models["price_predictor"].predict(pd.DataFrame([item.model_dump()]))[0][0]}

@app.post("/predict_items_from_csv")
def predict_items_from_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    df['predicted_price'] = ml_models["price_predictor"].predict(df)
    file_result_name = 'data_with_predictions.csv'
    df.to_csv(file_result_name, index=False)

    return FileResponse(file_result_name, media_type='text/csv', filename=file_result_name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
