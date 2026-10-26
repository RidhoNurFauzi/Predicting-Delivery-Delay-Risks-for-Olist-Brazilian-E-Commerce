from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Logistics Late Delivery Prediction API")

artifacts = joblib.load('Model/late_delivery_model.pkl')
model = artifacts['model']
threshold = artifacts['threshold']
encodings = joblib.load('Model/encoding_maps.pkl')

class OrderInput(BaseModel):
    is_same_state: int
    freight_value: float
    product_weight_g: float
    product_volume_cm3: float
    price: float
    estimated_delivery_days: float
    allowed_processing_days: float
    estimated_courier_days: float
    is_weekend_purchase: int
    is_night_purchase: int
    freight_ratio: float
    product_density: float
    distance_km: float
    customer_state: str
    seller_state: str
    product_category: str

@app.post("/predict")
def predict_late_delivery(order: OrderInput):
    state_route = f"{order.customer_state}_{order.seller_state}"
    
    hist_route_risk = encodings['route'].get(state_route, encodings['prior'])
    hist_cat_risk = encodings['category'].get(order.product_category, encodings['prior'])
    
    route_vol_risk = hist_route_risk * order.product_volume_cm3
    route_weight_risk = hist_route_risk * order.product_weight_g
    
    features = {
        "is_same_state": order.is_same_state,
        "freight_value": order.freight_value,
        "product_weight_g": order.product_weight_g,
        "product_volume_cm3": order.product_volume_cm3,
        "price": order.price,
        "estimated_delivery_days": order.estimated_delivery_days,
        "allowed_processing_days": order.allowed_processing_days,
        "estimated_courier_days": order.estimated_courier_days,
        "is_weekend_purchase": order.is_weekend_purchase,
        "is_night_purchase": order.is_night_purchase,
        "freight_ratio": order.freight_ratio,
        "product_density": order.product_density,
        "distance_km": order.distance_km,
        "historical_route_risk": hist_route_risk,
        "historical_category_risk": hist_cat_risk,
        "route_volume_risk": route_vol_risk,
        "route_weight_risk": route_weight_risk
    }
    
    df_input = pd.DataFrame([features])
    
    prob_late = model.predict_proba(df_input)[:, 1][0]
    is_late_pred = int(prob_late >= threshold)
    
    return {
        "probability_late": round(float(prob_late), 4),
        "is_late_prediction": is_late_pred,
        "status": "High Risk" if is_late_pred == 1 else "Safe",
        "route_analyzed": state_route
    }