import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit as st
import base64
import joblib
from datetime import date

from app.predictor import MatchPredictor

st.set_page_config(
    page_title="Pitch Prophet · EPL Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="auto",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "best_model.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
TEAM_DF_PATH = MODELS_DIR / "team_df.pkl"
ELO_DF_PATH = MODELS_DIR / "df_with_elo.pkl"


# Outcome colours (used for the verdict and the probability bar)
OUTCOME_COLORS = {
    "Home Win": "#22c55e",
    "Draw": "#f5b301",
    "Away Win": "#ef4444",
}

# Badge colours, picked per team so each club always gets the same one
BADGE_COLORS = [
    "#e11d48", "#2563eb", "#7c3aed", "#0891b2", "#ea580c",
    "#16a34a", "#be123c", "#0f766e", "#9333ea", "#ca8a04",
]


def result_color(prediction: str) -> str:
    return OUTCOME_COLORS.get(prediction, "#ef4444")


def team_code(team: str) -> str:
    """Three-letter scoreboard code, e.g. 'man city' -> 'MCI', 'arsenal' -> 'ARS'."""
    words = [w for w in team.replace("'", "").split() if w]
    if len(words) == 1:
        return words[0][:3].upper()
    return (words[0][0] + words[-1][:2]).upper()


def badge_color(team: str) -> str:
    return BADGE_COLORS[sum(ord(c) for c in team) % len(BADGE_COLORS)]


def display_name(team: str) -> str:
    return " ".join(w.capitalize() for w in team.split())


# Styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@600;700;800&display=swap');

.stApp {
    background:
        repeating-linear-gradient(90deg, rgba(255,255,255,.025) 0 90px, transparent 90px 180px),
        radial-gradient(ellipse at 50% -10%, #1f6b45 0%, #0f3d2a 45%, #0a2a1d 100%);
    background-attachment: fixed;
    font-family: 'Barlow', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stDecoration"] { display: none; }

.block-container { max-width: 1050px; padding-top: 2.6rem; padding-bottom: 2rem; }

.stApp, .stApp p, .stApp label { color: #e8f3ec; }

/* Masthead */
.masthead { text-align: center; margin-bottom: 1.6rem; }
.masthead-kicker {
    display: inline-block;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    letter-spacing: .22em;
    font-size: .8rem;
    color: #a7f3c4;
    border: 1px solid rgba(167,243,196,.35);
    border-radius: 99px;
    padding: .3rem .9rem;
    margin-bottom: .8rem;
}
.masthead-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: clamp(2.4rem, 6vw, 3.8rem);
    line-height: 1;
    letter-spacing: .01em;
    color: #ffffff;
    text-transform: uppercase;
}
.masthead-title span { color: #f5b301; }
.masthead-sub { color: #9cc7ae; font-size: 1rem; margin-top: .5rem; }

/* Input panel */
.st-key-match_panel {
    background: rgba(5, 25, 17, .55);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 18px;
    padding: 1.3rem 1.4rem 1.1rem;
    backdrop-filter: blur(6px);
}
.vs-chip {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    color: #f5b301;
    text-align: center;
    padding-top: 1.75rem;
}
.st-key-match_panel label p { color: #a7d7bb !important; font-weight: 600; letter-spacing: .02em; }
.st-key-match_panel [data-baseweb="select"] > div,
.st-key-match_panel [data-baseweb="input"] {
    background: rgba(255,255,255,.96);
    border-radius: 10px;
}
.st-key-match_panel [data-baseweb="select"] *,
.st-key-match_panel input { color: #0b1f16 !important; -webkit-text-fill-color: #0b1f16 !important; font-weight: 600; }
.st-key-match_panel [data-testid="stDateInputField"] * { color: #0b1f16 !important; }

.st-key-predict_button button {
    background: #f5b301;
    border: none;
    color: #0b1f16;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 1.15rem;
    letter-spacing: .06em;
    text-transform: uppercase;
    border-radius: 10px;
    min-height: 48px;
    margin-top: .4rem;
    transition: transform .15s ease, box-shadow .15s ease;
}
.st-key-predict_button button p { color: #0b1f16 !important; font-weight: 800; }
.st-key-predict_button button:hover {
    background: #ffc727;
    color: #0b1f16;
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(245,179,1,.35);
}

/* Scoreboard */
.scoreboard {
    margin-top: 1.4rem;
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1420 100%);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 22px;
    padding: 1.6rem 1.6rem 1.4rem;
    box-shadow: 0 24px 50px rgba(0,0,0,.35);
    animation: board-in .45s ease-out;
}
@keyframes board-in {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.board-meta {
    display: flex;
    justify-content: space-between;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    letter-spacing: .14em;
    font-size: .78rem;
    color: #7f93a8;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.board-teams {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: 1rem;
}
.team { display: flex; flex-direction: column; align-items: center; gap: .55rem; text-align: center; }
.badge {
    width: 74px;
    height: 74px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 1.45rem;
    letter-spacing: .04em;
    color: #ffffff;
    border: 3px solid rgba(255,255,255,.85);
    box-shadow: 0 8px 18px rgba(0,0,0,.35);
}
.team-name {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: 1.45rem;
    color: #ffffff;
    line-height: 1.05;
}
.team-role { font-size: .75rem; letter-spacing: .14em; color: #7f93a8; text-transform: uppercase; }
.board-vs {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    color: #3b4a5c;
}

.verdict {
    margin: 1.5rem auto 1.3rem;
    width: fit-content;
    padding: .55rem 1.4rem;
    border-radius: 99px;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    letter-spacing: .06em;
    text-transform: uppercase;
    color: #0a1420;
}

/* One bar split into the three outcomes */
.prob-bar {
    display: flex;
    height: 16px;
    border-radius: 99px;
    overflow: hidden;
    background: #1c2a3a;
}
.prob-bar div { height: 100%; animation: grow .8s ease-out; transform-origin: left; }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
.prob-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    margin-top: .9rem;
    text-align: center;
}
.prob-num {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    line-height: 1;
}
.prob-label { font-size: .8rem; color: #9fb1c4; margin-top: .25rem; letter-spacing: .04em; }
.prob-cell.top .prob-label { color: #ffffff; font-weight: 600; }

.placeholder-board {
    margin-top: 1.4rem;
    border: 2px dashed rgba(255,255,255,.18);
    border-radius: 22px;
    padding: 2.4rem 1rem;
    text-align: center;
    color: #9cc7ae;
    font-size: 1rem;
}
.placeholder-board b {
    display: block;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.6rem;
    color: #ffffff;
    letter-spacing: .04em;
    margin-bottom: .2rem;
}

@media (max-width: 640px) {
    .badge { width: 56px; height: 56px; font-size: 1.1rem; }
    .team-name { font-size: 1.15rem; }
    .prob-num { font-size: 1.5rem; }
    .vs-chip { padding-top: 0; }
}
</style>
""",
    unsafe_allow_html=True,
)


# Stadium background (an original illustration stored next to this file).
# A dark overlay keeps the scoreboard and text readable on top of it.
BACKGROUND_PATH = BASE_DIR / "ui" / "stadium_bg.jpg"


@st.cache_data
def background_css(path):
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"""
<style>
.stApp {{
    background:
        linear-gradient(180deg, rgba(4,16,11,.55) 0%, rgba(4,16,11,.72) 45%, rgba(4,16,11,.88) 100%),
        url("data:image/jpeg;base64,{encoded}") center top / cover no-repeat fixed;
}}
</style>
"""


st.markdown(background_css(BACKGROUND_PATH), unsafe_allow_html=True)


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
        df_with_elo=df_with_elo,
    )
    return predictor, team_df


try:
    predictor, team_df = load_predictor()
    teams = sorted(team_df["team"].astype(str).str.strip().str.lower().unique())
except Exception as e:
    predictor = None
    teams = []
    st.error(f"Could not load model artifacts: {e}")


# Masthead
st.markdown(
    """
<div class="masthead">
<div class="masthead-kicker">PREMIER LEAGUE PREDICTOR · BY JESSE</div>
<div class="masthead-title">Pitch <span>Prophet</span></div>
<div class="masthead-sub">Pick a fixture and see how the model calls it.</div>
</div>
""",
    unsafe_allow_html=True,
)


# Match input panel
with st.container(key="match_panel"):
    if teams:
        c1, c2, c3, c4 = st.columns([5, 1, 5, 4])
        with c1:
            home_team = st.selectbox(
                "Home team", teams, index=0, format_func=display_name
            )
        with c2:
            st.markdown('<div class="vs-chip">VS</div>', unsafe_allow_html=True)
        with c3:
            away_team = st.selectbox(
                "Away team",
                teams,
                index=1 if len(teams) > 1 else 0,
                format_func=display_name,
            )
        with c4:
            match_date = st.date_input("Match date", value=date.today())

        predict_button = st.button(
            "Predict match", use_container_width=True, key="predict_button"
        )
    else:
        home_team = away_team = None
        match_date = None
        predict_button = False

result_placeholder = st.empty()

with result_placeholder.container():
    st.markdown(
        """
<div class="placeholder-board">
<b>Kick-off pending</b>
Choose two teams and press Predict match.
</div>
""",
        unsafe_allow_html=True,
    )


def render_scoreboard(home_team, away_team, match_date, prediction, home_win, draw, away_win):
    probs = {"Home Win": home_win, "Draw": draw, "Away Win": away_win}
    top = max(probs, key=probs.get)

    verdict_text = {
        "Home Win": f"{display_name(home_team)} to win",
        "Away Win": f"{display_name(away_team)} to win",
        "Draw": "Draw",
    }.get(prediction, prediction)

    bar = "".join(
        f'<div style="width:{p * 100:.2f}%;background:{OUTCOME_COLORS[k]}"></div>'
        for k, p in probs.items()
    )

    cells = "".join(
        f'<div class="prob-cell{" top" if k == top else ""}">'
        f'<div class="prob-num" style="color:{OUTCOME_COLORS[k]}">{p * 100:.1f}%</div>'
        f'<div class="prob-label">{label}</div></div>'
        for (k, p), label in zip(
            probs.items(),
            [f"{display_name(home_team)} win", "Draw", f"{display_name(away_team)} win"],
        )
    )

    return f"""
<div class="scoreboard">
<div class="board-meta"><span>Premier League</span><span>{match_date:%a %d %b %Y}</span></div>
<div class="board-teams">
<div class="team">
<div class="badge" style="background:{badge_color(home_team)}">{team_code(home_team)}</div>
<div class="team-name">{display_name(home_team)}</div>
<div class="team-role">Home</div>
</div>
<div class="board-vs">VS</div>
<div class="team">
<div class="badge" style="background:{badge_color(away_team)}">{team_code(away_team)}</div>
<div class="team-name">{display_name(away_team)}</div>
<div class="team-role">Away</div>
</div>
</div>
<div class="verdict" style="background:{result_color(prediction)}">{verdict_text}</div>
<div class="prob-bar">{bar}</div>
<div class="prob-row">{cells}</div>
</div>
"""


# ---------------------------------------------------------------------------
# DEMO: "Why this prediction" and "Ask the Prophet"
# The factor breakdown below is illustrative. It is shaped to agree with the
# model's probabilities but is not computed from the model itself. Swap
# demo_factors() for real SHAP values when they are available.
# ---------------------------------------------------------------------------

def _seed(*parts):
    return sum(ord(c) for c in "".join(parts))


def demo_factors(home_team, away_team, home_win, away_win):
    """Illustrative factor contributions in percentage points.
    Positive values favour the home team, negative values favour the away team."""
    edge = (home_win - away_win) * 100
    home_advantage = 4.0
    remaining = edge - home_advantage

    jitter = (_seed(home_team, away_team) % 21 - 10) / 100  # -0.10 .. +0.10
    elo_share = 0.55 + jitter
    form_share = 0.30 - jitter / 2
    h2h_share = 1 - elo_share - form_share

    factors = [
        ("Elo rating gap", remaining * elo_share),
        ("Recent form (last 5)", remaining * form_share),
        ("Home advantage", home_advantage),
        ("Head to head record", remaining * h2h_share),
    ]
    return sorted(factors, key=lambda f: abs(f[1]), reverse=True)


def confidence_label(top_prob):
    if top_prob >= 0.60:
        return "high"
    if top_prob >= 0.45:
        return "moderate"
    return "low"


def render_factors(state, with_heading=True):
    home, away = display_name(state["home"]), display_name(state["away"])
    factors = state["factors"]
    biggest = max(abs(v) for _, v in factors) or 1

    rows = ""
    for name, value in factors:
        favours = home if value >= 0 else away
        colour = badge_color(state["home"]) if value >= 0 else badge_color(state["away"])
        width = abs(value) / biggest * 50
        side = "left:50%" if value >= 0 else f"right:50%"
        rows += f"""
<div class="factor">
<div class="factor-name">{name}</div>
<div class="factor-track"><div class="factor-mid"></div>
<div class="factor-fill" style="{side};width:{width:.1f}%;background:{colour}"></div></div>
<div class="factor-value">{value:+.1f} pts <span>→ {favours}</span></div>
</div>"""

    heading = (
        '<div class="panel-head"><span>Why this prediction</span>'
        '<span class="demo-tag">Demo</span></div>'
        if with_heading else ""
    )
    return f"""
<div class="panel{'' if with_heading else ' panel-flat'}">
{heading}
<div class="factor-axis"><span>← favours {away}</span><span>favours {home} →</span></div>
{rows}
<div class="panel-note">Illustrative breakdown for the demo. The live version will use SHAP values from the model.</div>
</div>
"""


def phrase(name):
    """Lower-case a factor name for use mid-sentence, keeping 'Elo' capitalised."""
    return name if name.startswith("Elo") else name[0].lower() + name[1:]


def prophet_reply(question, state):
    q = question.lower()
    home, away = display_name(state["home"]), display_name(state["away"])
    hw, dr, aw = state["home_win"], state["draw"], state["away_win"]
    probs = {f"{home} win": hw, "a draw": dr, f"{away} win": aw}
    top_name = max(probs, key=probs.get)
    top_prob = probs[top_name]
    factors = state["factors"]
    lead_name, lead_value = factors[0]
    lead_side = home if lead_value >= 0 else away

    if any(w in q for w in ["why", "explain", "reason", "because"]):
        parts = ", ".join(
            f"{phrase(n)} ({v:+.1f} pts)" for n, v in factors[:3]
        )
        if top_name == "a draw":
            return (
                f"I lean towards a draw at {top_prob:.0%}. The {phrase(lead_name)} tilts "
                f"things towards {lead_side}, but not by enough to make a win more likely "
                f"than a draw. The top factors are {parts}."
            )
        return (
            f"I lean towards {top_name} at {top_prob:.0%}. The biggest driver is the "
            f"{phrase(lead_name)}, which pushes things towards {lead_side}. "
            f"The top factors are {parts}. Positive points favour {home}, negative favour {away}."
        )
    if any(w in q for w in ["sure", "confident", "certain", "trust"]):
        level = confidence_label(top_prob)
        return (
            f"My confidence is {level}. My top pick, {top_name}, sits at {top_prob:.0%}, "
            f"so roughly {1 - top_prob:.0%} of the time something else happens. "
            "Football is noisy, so treat this as a lean, not a guarantee."
        )
    if "draw" in q:
        return (
            f"I give a draw {dr:.0%}. "
            + ("That's my most likely outcome here." if dr == max(hw, dr, aw)
               else "It's possible, but not my main pick.")
        )
    if any(w in q for w in ["upset", "underdog", "surprise"]):
        underdog, u_prob = (away, aw) if hw >= aw else (home, hw)
        verdict = "a live upset chance" if u_prob >= 0.25 else "a long shot"
        return f"The underdog is {underdog} at {u_prob:.0%}, which I'd call {verdict}."
    if any(w in q for w in ["change", "what if", "flip", "swing"]):
        return (
            f"The factor with the most weight is the {phrase(lead_name)}. If that "
            f"shifted the other way, the prediction would move most. Smaller factors "
            "like head to head would only nudge it."
        )
    if any(w in q for w in ["score", "goals", "scoreline"]):
        return (
            "I only predict the result (win, draw or loss) for now. "
            "A scoreline model is on the roadmap."
        )
    if any(w in q for w in ["bet", "odds", "gamble"]):
        return (
            "I'm a portfolio project, not betting advice. My numbers are "
            "probabilities from historical data, so please don't stake money on them."
        )
    return (
        f"I'm currently backing {top_name} ({top_prob:.0%}). Try asking why, "
        "how confident I am, about the draw, or whether an upset is likely."
    )


# Extra styling for the demo panels
st.markdown(
    """
<style>
.panel {
    margin-top: 1.1rem;
    background: rgba(5, 25, 17, .6);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 18px;
    padding: 1.2rem 1.3rem;
}
.panel-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 1.35rem;
    color: #ffffff;
    letter-spacing: .03em;
    margin-bottom: .6rem;
}
.demo-tag {
    font-size: .7rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #0b1f16;
    background: #a7f3c4;
    border-radius: 99px;
    padding: .2rem .6rem;
}
.factor-axis {
    display: flex;
    justify-content: space-between;
    font-size: .75rem;
    color: #8fb9a1;
    margin: 0 0 .5rem 0;
}
.factor {
    display: grid;
    grid-template-columns: 170px 1fr 170px;
    align-items: center;
    gap: .8rem;
    padding: .45rem 0;
    border-top: 1px solid rgba(255,255,255,.06);
}
.factor-name { color: #e8f3ec; font-weight: 600; font-size: .92rem; }
.factor-track { position: relative; height: 12px; background: rgba(255,255,255,.06); border-radius: 99px; }
.factor-mid { position: absolute; left: 50%; top: -3px; bottom: -3px; width: 2px; background: rgba(255,255,255,.35); }
.factor-fill { position: absolute; top: 0; bottom: 0; border-radius: 99px; }
.factor-value { color: #ffffff; font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.05rem; }
.factor-value span { color: #8fb9a1; font-family: 'Barlow', sans-serif; font-weight: 500; font-size: .8rem; }
.panel-note { color: #7fa892; font-size: .78rem; margin-top: .7rem; }

.st-key-prophet_chat {
    margin-top: 1.1rem;
    background: rgba(5, 25, 17, .6);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 18px;
    padding: 1.1rem 1.3rem 1rem;
}
.st-key-prophet_chat [data-testid="stChatMessage"] {
    background: rgba(255,255,255,.05);
    border-radius: 14px;
}
.st-key-prophet_chat [data-testid="stChatMessage"] p { color: #e8f3ec; }
.st-key-prophet_chat [data-baseweb="input"] { background: rgba(255,255,255,.96); border-radius: 10px; }
.st-key-prophet_chat input { color: #0b1f16 !important; -webkit-text-fill-color: #0b1f16 !important; }
.st-key-prophet_chat .stButton button {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.18);
    color: #e8f3ec;
    border-radius: 99px;
    font-size: .85rem;
}
.st-key-prophet_chat .stButton button p { color: #e8f3ec; }
.st-key-prophet_chat .stButton button:hover { border-color: #f5b301; color: #f5b301; }
.st-key-prophet_chat [data-testid="stFormSubmitButton"] button {
    background: #f5b301; border: none; border-radius: 10px;
}
.st-key-prophet_chat [data-testid="stFormSubmitButton"] button p { color: #0b1f16; font-weight: 700; }

/* Collapsible breakdown */
[data-testid="stExpander"] details {
    margin-top: 1rem;
    background: rgba(5, 25, 17, .6);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 16px;
}
[data-testid="stExpander"] summary p {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 1.2rem;
    letter-spacing: .03em;
    color: #ffffff;
}
[data-testid="stExpander"] summary svg { fill: #f5b301; color: #f5b301; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] details[open] > summary {
    background: transparent !important;
    border-radius: 16px;
}
[data-testid="stExpander"] summary:hover p { color: #f5b301; }

/* Wider side panel so the chat is comfortable to read */
section[data-testid="stSidebar"][aria-expanded="true"] {
    width: 360px !important;
    min-width: 360px !important;
}
.panel-flat { margin-top: 0; background: transparent; border: none; padding: 0; }
.chat-hint { text-align: center; color: #8fb9a1; font-size: .88rem; margin-top: .9rem; }
.chat-hint b { color: #f5b301; }

/* Sidebar chat */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1420 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #e8f3ec; }
.side-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 800;
    font-size: 1.9rem;
    text-transform: uppercase;
    color: #ffffff;
    line-height: 1;
}
.side-title span { color: #f5b301; }
.side-sub { color: #7f93a8; font-size: .8rem; margin: .35rem 0 1rem; }
.side-empty {
    border: 2px dashed rgba(255,255,255,.15);
    border-radius: 14px;
    padding: 1rem;
    color: #9fb1c4;
    font-size: .9rem;
    line-height: 1.5;
}
.st-key-prophet_messages { border-color: rgba(255,255,255,.1) !important; border-radius: 14px; }
[data-testid="stSidebar"] [data-testid="stChatMessage"] { background: rgba(255,255,255,.05); border-radius: 12px; }
[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.16);
    border-radius: 99px;
    min-height: 36px;
}
[data-testid="stSidebar"] .stButton button p { color: #e8f3ec; font-size: .85rem; }
[data-testid="stSidebar"] .stButton button:hover { border-color: #f5b301; }
[data-testid="stSidebar"] .stButton button:hover p { color: #f5b301; }
[data-testid="stSidebar"] [data-baseweb="input"] { background: rgba(255,255,255,.96); border-radius: 10px; }
[data-testid="stSidebar"] input { color: #0b1f16 !important; -webkit-text-fill-color: #0b1f16 !important; }
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button { background: #f5b301; border: none; border-radius: 10px; }
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button p { color: #0b1f16; font-weight: 700; }

@media (max-width: 640px) {
    .factor { grid-template-columns: 1fr; gap: .3rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# Run a new prediction and keep it in session state, so the chat and
# breakdown survive reruns when the user interacts with them
if predictor and teams and predict_button:
    if home_team == away_team:
        st.session_state.pop("match_state", None)
        result_placeholder.error("Home team and away team must be different.")
    else:
        try:
            with st.spinner("Crunching Elo ratings and team stats…"):
                result = predictor.predict(
                    home_team=home_team,
                    away_team=away_team,
                    match_date=str(match_date),
                )

            prediction = result["prediction"]
            probs = result["probabilities"]

            home_win = float(probs["Home Win"])
            draw = float(probs["Draw"])
            away_win = float(probs["Away Win"])

            st.session_state["match_state"] = {
                "home": home_team,
                "away": away_team,
                "date": match_date,
                "prediction": prediction,
                "home_win": home_win,
                "draw": draw,
                "away_win": away_win,
                "factors": demo_factors(home_team, away_team, home_win, away_win),
            }
            st.session_state["chat"] = []

        except Exception as e:
            st.session_state.pop("match_state", None)
            result_placeholder.error(f"Prediction failed: {e}")


state = st.session_state.get("match_state")

if state:
    result_placeholder.markdown(
        render_scoreboard(
            state["home"], state["away"], state["date"],
            state["prediction"], state["home_win"], state["draw"], state["away_win"],
        ),
        unsafe_allow_html=True,
    )

    # Breakdown folds away under the scoreboard to keep the page short
    with st.expander("Why this prediction  ·  demo", expanded=False):
        st.markdown(render_factors(state, with_heading=False), unsafe_allow_html=True)

    st.markdown(
        '<div class="chat-hint">Questions about this match? Open '
        '<b>Ask the Prophet</b> in the side panel.</div>',
        unsafe_allow_html=True,
    )


# Ask the Prophet lives in the collapsible sidebar
with st.sidebar:
    st.markdown(
        '<div class="side-title">Ask the <span>Prophet</span></div>'
        '<div class="side-sub">Demo chat · answers from the current prediction</div>',
        unsafe_allow_html=True,
    )

    if not state:
        st.markdown(
            '<div class="side-empty">Predict a match first, then ask me why I '
            "called it that way, how confident I am, or whether an upset is on.</div>",
            unsafe_allow_html=True,
        )
    else:
        chat = st.session_state.setdefault("chat", [])

        if not chat:
            chat.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Ask me anything about {display_name(state['home'])} vs "
                        f"{display_name(state['away'])}. I'll answer from this prediction."
                    ),
                }
            )

        # Scrollable message area, filled after the inputs are read
        messages_area = st.container(height=380, key="prophet_messages")

        asked = None
        for label in ["Why this prediction?", "How confident are you?", "Any upset chance?"]:
            if st.button(label, key=f"quick_{label}", use_container_width=True):
                asked = label

        with st.form("prophet_form", clear_on_submit=True, border=False):
            typed = st.text_input(
                "Your question",
                placeholder="e.g. What would change your mind?",
                label_visibility="collapsed",
            )
            sent = st.form_submit_button("Ask", use_container_width=True)

        if sent and typed.strip():
            asked = typed.strip()

        if asked:
            chat.append({"role": "user", "content": asked})
            chat.append({"role": "assistant", "content": prophet_reply(asked, state)})

        with messages_area:
            for message in chat:
                avatar = "🔮" if message["role"] == "assistant" else "🧑"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])