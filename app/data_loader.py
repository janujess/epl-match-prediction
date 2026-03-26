import pandas as pd


class MatchDataLoader:

    def __init__(self, db_connector):
        self.db_connector = db_connector

    def load_matches(self, table_name):

        query = f"SELECT * FROM {table_name}"

        conn = self.db_connector.get_connection()

        try:
            df = pd.read_sql(query, conn)

        finally:
            conn.close()

        return df