import pandas as pd
import numpy as np


class MatchPredictor:

    def __init__(self, model, feature_columns, team_df_model, df_with_elo):
        self.model = model
        self.feature_columns = list(feature_columns)
        self.team_df_model = team_df_model
        self.df_with_elo = df_with_elo

    @staticmethod
    def standardize_team_name(team_name: str) -> str:
        return team_name.strip().lower()

    def predict(self, home_team: str, away_team: str, match_date: str) -> dict:
        home_team = self.standardize_team_name(home_team)
        away_team = self.standardize_team_name(away_team)
        match_date = pd.Timestamp(match_date)

        valid_teams = set(
            self.team_df_model["team"].astype(str).str.strip().str.lower().unique()
        )

        if home_team not in valid_teams:
            raise ValueError(f"Home team '{home_team}' not found in dataset.")
        if away_team not in valid_teams:
            raise ValueError(f"Away team '{away_team}' not found in dataset.")
        if home_team == away_team:
            raise ValueError("Home team and away team cannot be the same.")

        home_hist = (
            self.team_df_model[
                (self.team_df_model["team"] == home_team) &
                (self.team_df_model["date"] < match_date)
            ]
            .sort_values("date")
        )

        away_hist = (
            self.team_df_model[
                (self.team_df_model["team"] == away_team) &
                (self.team_df_model["date"] < match_date)
            ]
            .sort_values("date")
        )

        if home_hist.empty:
            raise ValueError(f"No historical data found for {home_team} before {match_date.date()}.")
        if away_hist.empty:
            raise ValueError(f"No historical data found for {away_team} before {match_date.date()}.")

        home_latest = home_hist.iloc[-1]
        away_latest = away_hist.iloc[-1]

        home_elo_hist_home = self.df_with_elo[
            (self.df_with_elo["hometeam"] == home_team) &
            (self.df_with_elo["date"] < match_date)
        ][["date", "home_elo_pre"]].rename(columns={"home_elo_pre": "elo_pre"})

        home_elo_hist_away = self.df_with_elo[
            (self.df_with_elo["awayteam"] == home_team) &
            (self.df_with_elo["date"] < match_date)
        ][["date", "away_elo_pre"]].rename(columns={"away_elo_pre": "elo_pre"})

        away_elo_hist_home = self.df_with_elo[
            (self.df_with_elo["hometeam"] == away_team) &
            (self.df_with_elo["date"] < match_date)
        ][["date", "home_elo_pre"]].rename(columns={"home_elo_pre": "elo_pre"})

        away_elo_hist_away = self.df_with_elo[
            (self.df_with_elo["awayteam"] == away_team) &
            (self.df_with_elo["date"] < match_date)
        ][["date", "away_elo_pre"]].rename(columns={"away_elo_pre": "elo_pre"})

        home_elo_hist = pd.concat([home_elo_hist_home, home_elo_hist_away]).sort_values("date")
        away_elo_hist = pd.concat([away_elo_hist_home, away_elo_hist_away]).sort_values("date")

        if home_elo_hist.empty:
            raise ValueError(f"No Elo history found for {home_team} before {match_date.date()}.")
        if away_elo_hist.empty:
            raise ValueError(f"No Elo history found for {away_team} before {match_date.date()}.")

        home_elo = float(home_elo_hist.iloc[-1]["elo_pre"])
        away_elo = float(away_elo_hist.iloc[-1]["elo_pre"])

        feature_row = {}

        for col in self.feature_columns:
            if col == "elo_diff_pre":
                feature_row[col] = float(home_elo - away_elo)

            elif col.startswith("diff_"):
                base_col = col.replace("diff_", "", 1)

                if base_col not in home_latest.index:
                    raise KeyError(f"Required feature '{base_col}' missing for home team.")
                if base_col not in away_latest.index:
                    raise KeyError(f"Required feature '{base_col}' missing for away team.")

                feature_row[col] = float(home_latest[base_col] - away_latest[base_col])

            else:
                raise KeyError(f"Unexpected training column '{col}'.")

        X_match = pd.DataFrame([feature_row])
        X_match = X_match[self.feature_columns]

        pred_proba = np.asarray(self.model.predict_proba(X_match)).reshape(-1)
        pred_class = np.asarray(self.model.predict(X_match)).reshape(-1)[0]

        label_map = {
            0: "Home Win",
            1: "Draw",
            2: "Away Win"
        }

        return {
            "home_team": home_team,
            "away_team": away_team,
            "match_date": str(match_date.date()),
            "prediction": label_map[int(pred_class)],
            "probabilities": {
                "Home Win": float(pred_proba[0]),
                "Draw": float(pred_proba[1]),
                "Away Win": float(pred_proba[2])
            }
        }