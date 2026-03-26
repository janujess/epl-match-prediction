import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT"))
    MYSQL_USER = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
    MYSQL_TABLE = os.getenv("MYSQL_TABLE")

    # feature engineering settings
    LAGS = [1, 2, 3, 5]
    WINDOWS = [3, 5, 10]

    RANDOM_STATE = 42