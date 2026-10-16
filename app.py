from fastapi import FastAPI, HTTPException, Header, Depends, Response
from pydantic import BaseModel
from typing import List, Optional
import hmac
import logging
import os
import uvicorn
from recommender import Recommender

app = FastAPI()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("recommendation-service")

# The shared secret must come from the environment. It used to be a literal in this
# file, so it was in the repository and in every clone, and it could not be rotated
# without a code change. Refuse to start rather than fall back to a published value.
MIN_KEY_LENGTH = 32
API_KEY = os.environ.get("ML_SERVICE_KEY", "").strip()
if len(API_KEY) < MIN_KEY_LENGTH:
    raise SystemExit(
        "ML_SERVICE_KEY is missing or shorter than {} characters. Generate one with "
        "'openssl rand -base64 32' and set it on both this service and the API, which "
        "sends it as the X-ML-KEY header.".format(MIN_KEY_LENGTH)
    )

recommender = None
recommender_error = None


@app.on_event("startup")
def startup_event():
    global recommender, recommender_error
    try:
        recommender = Recommender()
        recommender_error = None
    except Exception as e:
        # Staying up lets the service report why it is broken instead of crash-looping,
        # but it must not claim to be healthy while every endpoint answers 503.
        recommender_error = str(e)
        log.exception("Failed to initialize recommender")


async def verify_key(x_ml_key: str = Header(None)):
    if not hmac.compare_digest(x_ml_key or "", API_KEY):
        raise HTTPException(status_code=403, detail="Unauthorized")
    return x_ml_key


@app.get("/health")
def health_check(response: Response):
    # Previously always 200, so a service whose model failed to load looked healthy
    # to every monitor while answering 503 to all of its real endpoints.
    if recommender is None:
        response.status_code = 503
        return {
            "status": "unavailable",
            "service": "car-recommendation-service",
            "reason": "recommender not initialized",
        }
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
