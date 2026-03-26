import streamlit as st
import joblib
from pathlib import Path
from datetime import date

from app.predictor import MatchPredictor

st.set_page_config(
    page_title="EPL Match Predictor",
    page_icon="⚽",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "best_model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
TEAM_DF_PATH = MODELS_DIR / "team_df.pkl"
ELO_DF_PATH = MODELS_DIR / "df_with_elo.pkl"


def result_color(prediction: str) -> str:
    if prediction == "Home Win":
        return "#16a34a"
    if prediction == "Draw":
        return "#d97706"
    return "#dc2626"


@st.cache_resource
def load_predictor():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    team_df = joblib.load(TEAM_DF_PATH)
    df_with_elo = joblib.load(ELO_DF_PATH)

    predictor = MatchPredictor(
        model=model,
        feature_columns=feature_columns,
        team_df_model=team_df,
        df_with_elo=df_with_elo
    )
    return predictor, team_df


try:
    predictor, team_df = load_predictor()
    teams = sorted(team_df["team"].astype(str).str.strip().str.lower().unique())
except Exception as e:
    predictor = None
    teams = []
    st.error(f"Could not load model artifacts: {e}")

st.markdown("## JESSE`S EPL Match Predictor")
st.markdown("Choose teams and a date, then view the prediction instantly.")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Match Input")

    if teams:
        home_team = st.selectbox("Home Team", teams, index=0)
        away_team = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0)
        match_date = st.date_input("Match Date", value=date.today())
        predict_button = st.button("Predict Match", use_container_width=True)
    else:
        home_team = away_team = None
        match_date = None
        predict_button = False

with right_col:
    st.subheader("Prediction Result")
    result_placeholder = st.empty()

    with result_placeholder.container():
        st.info("Prediction output will appear here.")

if predictor and teams and predict_button:
    if home_team == away_team:
        with right_col:
            result_placeholder.error("Home team and away team must be different.")
    else:
        try:
            result = predictor.predict(
                home_team=home_team,
                away_team=away_team,
                match_date=str(match_date)
            )

            prediction = result["prediction"]
            probs = result["probabilities"]

            home_win = float(probs["Home Win"])
            draw = float(probs["Draw"])
            away_win = float(probs["Away Win"])

            banner_color = result_color(prediction)

            with right_col:
                result_placeholder.empty()
                with result_placeholder.container():
                    st.markdown(
                        f"""
                        <div style="
                            background:{banner_color};
                            padding:14px;
                            border-radius:10px;
                            color:white;
                            font-weight:700;
                            text-align:center;
                            font-size:22px;
                        ">
                            {prediction}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(f"### {home_team.title()} vs {away_team.title()}")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Home", f"{home_win * 100:.1f}%")
                    m2.metric("Draw", f"{draw * 100:.1f}%")
                    m3.metric("Away", f"{away_win * 100:.1f}%")

                    st.write("Home Win")
                    st.progress(home_win)

                    st.write("Draw")
                    st.progress(draw)

                    st.write("Away Win")
                    st.progress(away_win)

        except Exception as e:
            with right_col:
                result_placeholder.error(f"Prediction failed: {e}")