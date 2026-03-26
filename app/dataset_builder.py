import pandas as pd


class DatasetBuilder:

    def build_match_dataset(self, clean_df: pd.DataFrame, team_df: pd.DataFrame, df_with_elo: pd.DataFrame) -> pd.DataFrame:
        team_df_model = team_df.dropna(
            subset=['points_roll_mean_5', 'gf_roll_mean_5', 'ga_roll_mean_5']
        ).copy()

        home_features = team_df_model[team_df_model['home'] == 1].copy()
        away_features = team_df_model[team_df_model['home'] == 0].copy()

        home_features = home_features.add_prefix('home_')
        home_features.rename(columns={
            'home_date': 'date',
            'home_team': 'hometeam',
            'home_opponent': 'awayteam'
        }, inplace=True)

        away_features = away_features.add_prefix('away_')
        away_features.rename(columns={
            'away_date': 'date',
            'away_team': 'awayteam',
            'away_opponent': 'hometeam'
        }, inplace=True)

        match_df = pd.merge(
            home_features,
            away_features,
            on=['date', 'hometeam', 'awayteam'],
            how='inner'
        )

        match_results = clean_df[['date', 'hometeam', 'awayteam', 'ftr']].copy()

        match_df = match_df.merge(
            match_results,
            on=['date', 'hometeam', 'awayteam'],
            how='left'
        )

        elo_cols = [
            'date',
            'hometeam',
            'awayteam',
            'home_elo_pre',
            'away_elo_pre',
            'elo_diff_pre'
        ]

        match_df = match_df.merge(
            df_with_elo[elo_cols],
            on=['date', 'hometeam', 'awayteam'],
            how='left'
        )

        result_map = {'H': 0, 'D': 1, 'A': 2}
        match_df['target'] = match_df['ftr'].map(result_map)

        match_df = match_df.dropna(
            subset=[
                'home_points_roll_mean_10',
                'away_points_roll_mean_10'
            ]
        ).copy()

        safe_feature_cols = [
            c for c in match_df.columns
            if (('_lag_' in c) or ('_roll_' in c) or ('win_rate' in c) or ('_ewm_' in c))
        ]

        model_df = match_df[
            ['date', 'hometeam', 'awayteam', 'target', 'elo_diff_pre'] + safe_feature_cols
        ].copy()

        diff_features = {}

        for c in safe_feature_cols:
            if c.startswith('home_'):
                away_c = c.replace('home_', 'away_', 1)
                if away_c in model_df.columns:
                    diff_features[c.replace('home_', 'diff_', 1)] = model_df[c] - model_df[away_c]

        diff_df = pd.DataFrame(diff_features, index=model_df.index)

        final_df = pd.concat(
            [
                model_df[['date', 'hometeam', 'awayteam', 'target', 'elo_diff_pre']],
                diff_df
            ],
            axis=1
        )

        return final_df

    def split_data(self, final_df: pd.DataFrame) -> dict:
        split_date = final_df['date'].quantile(0.8)

        train = final_df[final_df['date'] <= split_date].copy()
        test = final_df[final_df['date'] > split_date].copy()

        split_date_train = train['date'].quantile(0.8)

        train_inner = train[train['date'] <= split_date_train].copy()
        valid_inner = train[train['date'] > split_date_train].copy()

        X_train = train_inner.drop(columns=['date', 'hometeam', 'awayteam', 'target']).copy()
        y_train = train_inner['target'].astype(int).copy()

        X_valid = valid_inner.drop(columns=['date', 'hometeam', 'awayteam', 'target']).copy()
        y_valid = valid_inner['target'].astype(int).copy()

        X_test = test.drop(columns=['date', 'hometeam', 'awayteam', 'target']).copy()
        y_test = test['target'].astype(int).copy()

        X_train = X_train.astype(float)
        X_valid = X_valid.astype(float)
        X_test = X_test.astype(float)

        return {
            "train_df": train_inner,
            "valid_df": valid_inner,
            "test_df": test,
            "X_train": X_train,
            "y_train": y_train,
            "X_valid": X_valid,
            "y_valid": y_valid,
            "X_test": X_test,
            "y_test": y_test
        }