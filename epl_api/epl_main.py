from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import traceback

from app.predictor import MatchPredictor

app = FastAPI(title="EPL Prediction API")

model = joblib.load("C:/Users/kipch/PycharmProjects/epl_prediction/models/xgb_weighted.pkl")
feature_columns = joblib.load("C:/Users/kipch/PycharmProjects/epl_prediction/models/feature_columns.pkl")
team_df_model = joblib.load("C:/Users/kipch/PycharmProjects/epl_prediction/models/team_df.pkl")
df_with_elo = joblib.load("C:/Users/kipch/PycharmProjects/epl_prediction/models/df_with_elo.pkl")

predictor = MatchPredictor(
    model=model,
    feature_columns=feature_columns,
    team_df_model=team_df_model,
    df_with_elo=df_with_elo
)

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    match_date: str

@app.get("/")
def root():
    return {"message": "EPL Prediction API running"}

@app.get("/teams")
def get_teams():
    teams = sorted(
        team_df_model["team"].astype(str).str.strip().str.lower().unique().tolist()
    )
    return {"teams": teams}

@app.post("/predict")
def predict_match(request: MatchRequest):
    try:
        result = predictor.predict(
            home_team=request.home_team,
            away_team=request.away_team,
            match_date=request.match_date
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Feature error: {str(e)}")

    except Exception as e:
        print("FULL TRACEBACK:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")