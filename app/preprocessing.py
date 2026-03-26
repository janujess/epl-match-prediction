import pandas as pd


class MatchPreprocessor:

    COLS_TO_KEEP = [
        'match_date','HomeTeam','AwayTeam','FTHG','FTAG','FTR',
        'HTHG','HTAG','HTR','Referee','HS','AS','HST','AST',
        'HC','AC','HF','AF','HY','AY','HR','AR'
    ]

    CAT_COLS = ['hometeam','awayteam','referee','ftr','htr']

    NUM_COLS = [
        'fthg','ftag','hthg','htag','hs','as','hst','ast',
        'hc','ac','hf','af','hy','ay','hr','ar'
    ]

    def clean(self, df):

        df = df.copy()

        # keep relevant columns
        df = df[self.COLS_TO_KEEP]

        # standardize column names
        df.columns = df.columns.str.strip().str.lower()

        # drop missing rows
        df = df.dropna(axis=0)

        # clean categorical text
        for col in ['hometeam','awayteam','referee']:
            df[col] = df[col].str.strip().str.lower()

        for col in ['ftr','htr']:
            df[col] = df[col].str.strip()

        # convert types
        for col in self.CAT_COLS:
            df[col] = df[col].astype('category')

        for col in self.NUM_COLS:
            df[col] = df[col].astype(int)

        # convert date
        df['date'] = pd.to_datetime(df['match_date'], errors='coerce', dayfirst=True)

        # remove bad rows
        df = df.dropna(subset=['date'])

        # remove duplicates
        df = df.drop_duplicates(subset=['date','hometeam','awayteam'])

        # sort
        df = df.sort_values('date').reset_index(drop=True)

        return df