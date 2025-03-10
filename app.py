from contextlib import asynccontextmanager
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
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

@app.get("/")
async def root():
    return {
        "Name": "Cars Prediction",
        "description": "This is a cars' price prediction model based on their typical features.",
    }

def medinc_regressor(x: dict) -> dict:
    with open("model.pkl", 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    x_df = pd.DataFrame(x, index=[0])
    res = loaded_model.predict(x_df)[0]
    return {"prediction": res}

ml_models = {}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["medinc_regressor"] = medinc_regressor
    yield
    ml_models.clear()

app = FastAPI(lifespan=ml_lifespan_manager)

@app.post("/predict_item")
async def predict(item: Item):
    return ml_models["medinc_regressor"](item.model_dump())

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return [ml_models["medinc_regressor"](item.model_dump()) for item in items]
