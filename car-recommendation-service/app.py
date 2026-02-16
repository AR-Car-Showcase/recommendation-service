from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from recommender import Recommender

app = FastAPI()

API_KEY = "carshowcase_internal_2026"
recommender = None


@app.on_event("startup")
def startup_event():
    global recommender
    try:
        recommender = Recommender()
    except Exception as e:
        print(f"Failed to initialize recommender: {e}")


async def verify_key(x_ml_key: str = Header(None)):
    if x_ml_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized")
    return x_ml_key


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "car-recommendation-service"}


@app.get("/recommend/similar/{car_model}")
def recommend_similar(car_model: str, x_ml_key: str = Depends(verify_key)):
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender not initialized")

    recommendations = recommender.get_similar_cars(car_model)

    if not recommendations:
        raise HTTPException(status_code=404, detail="Car not found or no recommendations")

    return recommendations


class UserPreferences(BaseModel):
    brands: List[str] = []
    body_types: List[str] = []
    fuel_types: List[str] = []
    transmissions: List[str] = []
    budget: float = 1000.0
    driving_condition: Optional[str] = None
    exclude_car_ids: List[str] = []
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@app.post("/recommend/personalized")
def recommend_personalized(preferences: UserPreferences, x_ml_key: str = Depends(verify_key)):
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender not initialized")

    pref_dict = preferences.dict()
    recommendations = recommender.get_personalized_ranking(pref_dict)

    return recommendations


class FeedbackRequest(BaseModel):
    car_id: str
    action: str


@app.post("/recommend/feedback")
def record_feedback(feedback: FeedbackRequest, x_ml_key: str = Depends(verify_key)):
    if not recommender:
        raise HTTPException(status_code=503, detail="Recommender not initialized")

    positive = feedback.action in ['view', 'click', 'like', 'compare']
    recommender.update_feedback(feedback.car_id, positive)

    return {"status": "recorded", "car_id": feedback.car_id, "action": feedback.action, "positive": positive}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
