import joblib


class ModelIO:

    @staticmethod
    def save_object(obj, file_path: str):
        joblib.dump(obj, file_path)

    @staticmethod
    def load_object(file_path: str):
        return joblib.load(file_path)