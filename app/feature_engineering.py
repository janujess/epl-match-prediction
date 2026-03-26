import numpy as np
import pandas as pd


class FeatureEngineer:

    def __init__(self, lags=None, windows=None):
        self.lags = lags or [1, 2, 3, 5]
        self.windows = windows or [3, 5, 10]

    def build_team_match_view(self, df: pd.DataFrame) -> pd.DataFrame:
        home_df = df[
            ['date', 'hometeam', 'awayteam', 'fthg', 'ftag', 'hs', 'as', 'hst', 'ast',
             'hc', 'ac', 'hf', 'af', 'hy', 'ay', 'hr', 'ar', 'ftr']
        ].copy()

        home_df.rename(columns={
            'hometeam': 'team',
            'awayteam': 'opponent',
            'fthg': 'goals_for',
            'ftag': 'goals_against',
            'hs': 'shots_for',
            'as': 'shots_against',
            'hst': 'shots_on_target_for',
            'ast': 'shots_on_target_against',
            'hc': 'corners_for',
            'ac': 'corners_against',
            'hf': 'fouls_for',
            'af': 'fouls_against',
            'hy': 'yellow_for',
            'ay': 'yellow_against',
            'hr': 'red_for',
            'ar': 'red_against'
        }, inplace=True)

        home_df['home'] = 1

        away_df = df[
            ['date', 'hometeam', 'awayteam', 'fthg', 'ftag', 'hs', 'as', 'hst', 'ast',
             'hc', 'ac', 'hf', 'af', 'hy', 'ay', 'hr', 'ar', 'ftr']
        ].copy()

        away_df.rename(columns={
            'awayteam': 'team',
            'hometeam': 'opponent',
            'ftag': 'goals_for',
            'fthg': 'goals_against',
            'as': 'shots_for',
            'hs': 'shots_against',
            'ast': 'shots_on_target_for',
            'hst': 'shots_on_target_against',
            'ac': 'corners_for',
            'hc': 'corners_against',
            'af': 'fouls_for',
            'hf': 'fouls_against',
            'ay': 'yellow_for',
            'hy': 'yellow_against',
            'ar': 'red_for',
            'hr': 'red_against'
        }, inplace=True)

        away_df['home'] = 0

        team_df = pd.concat([home_df, away_df], ignore_index=True)
        team_df = team_df.sort_values(['team', 'date']).reset_index(drop=True)

        team_df['goal_diff'] = team_df['goals_for'] - team_df['goals_against']
        team_df['result'] = team_df['goal_diff'].apply(
            lambda x: 'W' if x > 0 else ('D' if x == 0 else 'L')
        )

        points_map = {'W': 3, 'D': 1, 'L': 0}
        team_df['points'] = team_df['result'].map(points_map)

        return team_df

    def add_lag_features(self, team_df: pd.DataFrame) -> pd.DataFrame:
        team_df = team_df.copy()

        for lag in self.lags:
            team_df[f'points_lag_{lag}'] = team_df.groupby('team')['points'].shift(lag)
            team_df[f'gf_lag_{lag}'] = team_df.groupby('team')['goals_for'].shift(lag)
            team_df[f'ga_lag_{lag}'] = team_df.groupby('team')['goals_against'].shift(lag)

        return team_df

    def add_rolling_features(self, team_df: pd.DataFrame) -> pd.DataFrame:
        team_df = team_df.copy()

        for w in self.windows:
            team_df[f'points_roll_mean_{w}'] = (
                team_df.groupby('team')['points']
                .transform(lambda s: s.shift(1).rolling(window=w, min_periods=w).mean())
            )

            team_df[f'gf_roll_mean_{w}'] = (
                team_df.groupby('team')['goals_for']
                .transform(lambda s: s.shift(1).rolling(window=w, min_periods=w).mean())
            )

            team_df[f'ga_roll_mean_{w}'] = (
                team_df.groupby('team')['goals_against']
                .transform(lambda s: s.shift(1).rolling(window=w, min_periods=w).mean())
            )

        return team_df

    def add_form_features(self, team_df: pd.DataFrame) -> pd.DataFrame:
        team_df = team_df.copy()

        team_df['is_win'] = (team_df['result'] == 'W').astype(int)
        team_df['is_draw'] = (team_df['result'] == 'D').astype(int)
        team_df['is_loss'] = (team_df['result'] == 'L').astype(int)

        for w in self.windows:
            team_df[f'win_rate_{w}'] = (
                team_df.groupby('team')['is_win']
                .transform(lambda s: s.shift(1).rolling(window=w, min_periods=w).mean())
            )

        team_df['points_ewm_5'] = (
            team_df.groupby('team')['points']
            .transform(lambda s: s.shift(1).ewm(span=5, adjust=False).mean())
        )

        team_df['gd_ewm_5'] = (
            team_df.groupby('team')['goal_diff']
            .transform(lambda s: s.shift(1).ewm(span=5, adjust=False).mean())
        )

        return team_df

    def add_elo_features(
        self,
        matches_df: pd.DataFrame,
        k: int = 20,
        home_advantage: int = 60,
        base_elo: int = 1500,
        reset_factor: float = 0.75
    ):
        df_elo = matches_df.sort_values('date').copy().reset_index(drop=True)

        df_elo['season'] = np.where(
            df_elo['date'].dt.month >= 8,
            df_elo['date'].dt.year,
            df_elo['date'].dt.year - 1
        )

        teams = pd.unique(pd.concat([df_elo['hometeam'], df_elo['awayteam']], ignore_index=True))
        elo_dict = {team: base_elo for team in teams}

        home_elo_pre = []
        away_elo_pre = []
        elo_diff_pre = []

        current_season = df_elo.loc[0, 'season']

        for _, row in df_elo.iterrows():
            if row['season'] != current_season:
                for team in elo_dict:
                    elo_dict[team] = (
                        elo_dict[team] * reset_factor +
                        base_elo * (1 - reset_factor)
                    )
                current_season = row['season']

            home = row['hometeam']
            away = row['awayteam']
            result = row['ftr']

            home_elo = elo_dict[home]
            away_elo = elo_dict[away]

            home_elo_pre.append(home_elo)
            away_elo_pre.append(away_elo)
            elo_diff_pre.append(home_elo - away_elo)

            expected_home = 1 / (1 + 10 ** (-(home_elo + home_advantage - away_elo) / 400))
            expected_away = 1 - expected_home

            if result == 'H':
                actual_home, actual_away = 1.0, 0.0
            elif result == 'D':
                actual_home, actual_away = 0.5, 0.5
            else:
                actual_home, actual_away = 0.0, 1.0

            elo_dict[home] = home_elo + k * (actual_home - expected_home)
            elo_dict[away] = away_elo + k * (actual_away - expected_away)

        df_elo['home_elo_pre'] = home_elo_pre
        df_elo['away_elo_pre'] = away_elo_pre
        df_elo['elo_diff_pre'] = elo_diff_pre

        return df_elo, elo_dict