import streamlit as st
import requests
from datetime import date

API_BASE_URL = "http://127.0.0.1:8004"
PREDICT_URL = f"{API_BASE_URL}/predict"
TEAMS_URL = f"{API_BASE_URL}/teams"

st.set_page_config(
    page_title="EPL Match Predictor",
    page_icon="⚽",
    layout="wide"
)


def result_color(prediction: str) -> str:
    if prediction == "Home Win":
        return "#16a34a"
    if prediction == "Draw":
        return "#d97706"
    return "#dc2626"

@st.cache_data(ttl=300)
def load_teams():
    response = requests.get(TEAMS_URL, timeout=30)
    response.raise_for_status()
    return response.json()["teams"]

try:
    teams = load_teams()
except Exception:
    teams = []
    st.error("Could not load teams from the API. Make sure FastAPI is running.")

st.markdown('<div class="small-title">⚽ JESSE`S EPL Match Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Choose teams and a date, then view the prediction instantly.</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
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

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    result_placeholder = st.empty()

    with result_placeholder.container():
        st.info("Prediction output will appear here.")
    st.markdown('</div>', unsafe_allow_html=True)

if teams and predict_button:
    if home_team == away_team:
        with right_col:
            result_placeholder.error("Home team and away team must be different.")
    else:
        payload = {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": str(match_date)
        }

        try:
            response = requests.post(PREDICT_URL, json=payload, timeout=30)

            with right_col:
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    probs = result["probabilities"]

                    home_win = float(probs["Home Win"])
                    draw = float(probs["Draw"])
                    away_win = float(probs["Away Win"])

                    banner_color = result_color(prediction)

                    result_placeholder.empty()
                    with result_placeholder.container():
                        st.markdown(
                            f'<div class="pred-box" style="background:{banner_color};">{prediction}</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f'<div class="team-line">{home_team.title()} vs {away_team.title()}</div>',
                            unsafe_allow_html=True
                        )

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Home", f"{home_win * 100:.1f}%")
                        m2.metric("Draw", f"{draw * 100:.1f}%")
                        m3.metric("Away", f"{away_win * 100:.1f}%")

                        st.markdown('<div class="prob-label">Home Win</div>', unsafe_allow_html=True)
                        st.progress(home_win)

                        st.markdown('<div class="prob-label">Draw</div>', unsafe_allow_html=True)
                        st.progress(draw)

                        st.markdown('<div class="prob-label">Away Win</div>', unsafe_allow_html=True)
                        st.progress(away_win)

                else:
                    result_placeholder.error(f"API error: {response.status_code}")
                    try:
                        st.json(response.json())
                    except Exception:
                        st.write(response.text)

        except requests.exceptions.ConnectionError:
            with right_col:
                result_placeholder.error("Could not connect to the API. Make sure FastAPI is running.")
        except Exception as e:
            with right_col:
                result_placeholder.error(f"Unexpected error: {e}")