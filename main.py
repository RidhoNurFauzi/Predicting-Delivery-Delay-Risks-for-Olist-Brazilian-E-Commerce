from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Logistics Late Delivery Prediction API")

artifacts = joblib.load('Model/late_delivery_model.pkl')
model = artifacts['model']
threshold = artifacts['threshold']
encodings = joblib.load('Model/encoding_maps.pkl')


class OrderInput(BaseModel):
    freight_value: float = Field(ge=0)
    price: float = Field(gt=0)
    product_weight_g: float = Field(ge=0)
    product_volume_cm3: float = Field(ge=0)
    estimated_delivery_days: float
    allowed_processing_days: float
    distance_km: float
    customer_state: str
    seller_state: str
    product_category: str
    purchase_dayofweek: int = Field(ge=0, le=6)
    purchase_hour: int = Field(ge=0, le=23)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Logistics Late Delivery Prediction API"}


@app.post("/predict")
def predict_late_delivery(order: OrderInput):
    # Fitur kategorikal -> target encoding (kamus dari training)
    state_route = f"{order.customer_state}_{order.seller_state}"
    hist_route_risk = encodings['route'].get(state_route, encodings['prior'])
    hist_cat_risk = encodings['category'].get(order.product_category, encodings['prior'])

    # Fitur turunan (harus identik dengan notebook training)
    features = {
        "is_same_state": int(order.customer_state == order.seller_state),
        "freight_value": order.freight_value,
        "product_weight_g": order.product_weight_g,
        "product_volume_cm3": order.product_volume_cm3,
        "price": order.price,
        "estimated_delivery_days": order.estimated_delivery_days,
        "allowed_processing_days": order.allowed_processing_days,
        "estimated_courier_days": order.estimated_delivery_days - order.allowed_processing_days,
        "is_weekend_purchase": int(order.purchase_dayofweek in (5, 6)),
        "is_night_purchase": int(order.purchase_hour in (22, 23, 0, 1, 2, 3, 4, 5)),
        "freight_ratio": order.freight_value / (order.price + 0.001),
        "product_density": order.product_weight_g / (order.product_volume_cm3 + 0.001),
        "distance_km": order.distance_km,
        "historical_route_risk": hist_route_risk,
        "historical_category_risk": hist_cat_risk,
        "route_volume_risk": hist_route_risk * order.product_volume_cm3,
        "route_weight_risk": hist_route_risk * order.product_weight_g,
    }

    # Samakan urutan kolom dengan yang dilihat model saat training
    df_input = pd.DataFrame([features])[model.feature_name_]

    prob_late = model.predict_proba(df_input)[:, 1][0]
    is_late_pred = int(prob_late >= threshold)

    return {
        "probability_late": round(float(prob_late), 4),
        "is_late_prediction": is_late_pred,
        "status": "High Risk" if is_late_pred == 1 else "Safe",
        "route_analyzed": state_route,
    }