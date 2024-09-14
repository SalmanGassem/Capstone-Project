from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd 
import datetime
import joblib

app = FastAPI()

label_encoders = joblib.load("label_encoders.joblib")
model = joblib.load("USA_predict.joblib")

class CarData(BaseModel):
    manufacturer: str
    model: str
    year: int
    mileage: float
    mpg: float
    drivetrain: str
    accidents_or_damage: int

def calculate_age(year: int) -> int:
    current_year = datetime.datetime.now().year
    return current_year - year

@app.post("/predict/")
async def print(car: CarData):
    input_data = {
        'manufacturer': [car.manufacturer],
        'model': [car.model],
        'age': [calculate_age(car.year)],
        'mileage': [car.mileage],
        'mpg': [car.mpg],
        'drivetrain': [car.drivetrain],
        'accidents_or_damage': [car.accidents_or_damage]
    }
    input_df = pd.DataFrame(input_data)

    for feature in label_encoders:
        if feature in input_df.columns:
            input_df[feature] = label_encoders[feature].transform(input_df[feature])

    prediction = float(model.predict(input_df))
    return {'Prediction': prediction}
