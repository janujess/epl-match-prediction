from app.config import Config
from app.database import DatabaseConnector
from app.data_loader import MatchDataLoader
from app.preprocessing import MatchPreprocessor
from app.feature_engineering import FeatureEngineer
from app.dataset_builder import DatasetBuilder
from app.trainer import ModelTrainer
from app.evaluation import ModelEvaluator
from app.predictor import MatchPredictor
from app.model_io import ModelIO


def main():
    print("Starting pipeline...")

    db = DatabaseConnector(
        host=Config.MYSQL_HOST,
        port=Config.MYSQL_PORT,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DATABASE
    )

    loader = MatchDataLoader(db)
    preprocessor = MatchPreprocessor()
    feature_engineer = FeatureEngineer(
        lags=Config.LAGS,
        windows=Config.WINDOWS
    )
    dataset_builder = DatasetBuilder()
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    model_io = ModelIO()

   #data loading and preprocessing
    raw_df = loader.load_matches(Config.MYSQL_TABLE)
    print("Raw dataset shape:", raw_df.shape)

    clean_df = preprocessor.clean(raw_df)
    print("Clean dataset shape:", clean_df.shape)

    #feature engineering
    team_df = feature_engineer.build_team_match_view(clean_df)
    team_df = feature_engineer.add_lag_features(team_df)
    team_df = feature_engineer.add_rolling_features(team_df)
    team_df = feature_engineer.add_form_features(team_df)
    print("Team df after features:", team_df.shape)

    df_with_elo, final_elo_dict = feature_engineer.add_elo_features(clean_df)
    print("Elo df shape:", df_with_elo.shape)

    final_df = dataset_builder.build_match_dataset(clean_df, team_df, df_with_elo)
    print("Final modeling df shape:", final_df.shape)

    #data splitting
    splits = dataset_builder.split_data(final_df)

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_valid = splits["X_valid"]
    y_valid = splits["y_valid"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    print("X_train shape:", X_train.shape)
    print("X_valid shape:", X_valid.shape)
    print("X_test shape:", X_test.shape)

    #sample weights
    sample_weights_train, class_weight_dict = trainer.compute_sample_weights(y_train)
    print("Class weights:", class_weight_dict)

    #training baseline modes
    print("\nTraining Random Forest...")
    rf = trainer.train_random_forest(X_train, y_train)

    print("\nTraining XGBoost unweighted...")
    xgb_unweighted = trainer.train_xgb(
        X_train,
        y_train,
        X_valid,
        y_valid
    )

    print("\nTraining XGBoost weighted...")
    xgb_weighted = trainer.train_xgb(
        X_train,
        y_train,
        X_valid,
        y_valid,
        sample_weights=sample_weights_train
    )

    print("\nTraining LightGBM weighted...")
    lgbm = trainer.train_lgbm(
        X_train,
        y_train,
        X_valid,
        y_valid,
        sample_weights=sample_weights_train
    )
    #tuning  the models

    print("\nTuning XGBoost for F1_MACRO...")
    best_params_f1, best_cv_f1 = trainer.tune_xgb_search(
        X_train,
        y_train,
        scoring="f1_macro",
        n_splits=5,
        n_iter=200
    )

    print("\nBest params (f1_macro):")
    print(best_params_f1)
    print(f"Best CV f1_macro: {best_cv_f1:.4f}")

    print("\nFitting tuned XGBoost (f1_macro) with early stopping...")
    xgb_tuned_f1 = trainer.fit_best_xgb_with_early_stopping(
        X_train,
        y_train,
        X_valid,
        y_valid,
        best_params=best_params_f1,
        sample_weights=sample_weights_train
    )
    print("\nTraining CatBoost weighted...")
    catboost_model = trainer.train_catboost(
        X_train,
        y_train,
        X_valid,
        y_valid,
        sample_weights=sample_weights_train
    )
    # evaluating all models
    predictions = {
        "RandomForest": rf.predict(X_test),
        "XGBoost_Unweighted": xgb_unweighted.predict(X_test),
        "XGBoost_Weighted": xgb_weighted.predict(X_test),
        "LightGBM_Weighted": lgbm.predict(X_test),
        "XGBoost_Tuned_F1": xgb_tuned_f1.predict(X_test),
        "CatBoost_Weighted": catboost_model.predict(X_test).astype(int).ravel(),
    }

    results_df = evaluator.evaluate_models(
        y_test,
        predictions,
        labels=['Home_Win', 'Draw', 'Away_Win'],
        show_cm=True,
        show_cm_pct=True
    )

    print("\nResults summary:")
    print(results_df)
    #best model
    ranked_results = results_df.sort_values(
        by=["F1_macro", "Accuracy", "Precision_macro", "Recall_macro"],
        ascending=False
    ).reset_index(drop=True)

    best_model_name = ranked_results.loc[0, "Model"]
    print(f"\nBest overall model selected: {best_model_name}")

    model_lookup = {
        "RandomForest": rf,
        "XGBoost_Unweighted": xgb_unweighted,
        "XGBoost_Weighted": xgb_weighted,
        "LightGBM_Weighted": lgbm,
        "XGBoost_Tuned_F1": xgb_tuned_f1,
        'CatBoost_Weighted':catboost_model
    }

    best_model = model_lookup[best_model_name]

    #saving model artifacts
    model_io.save_object(xgb_weighted, "models/xgb_weighted.pkl")
    model_io.save_object(xgb_tuned_f1, "models/xgb_tuned_f1.pkl")
    model_io.save_object(best_model, "models/best_model.pkl")
    model_io.save_object(catboost_model, "models/catboost_weighted.pkl")
    model_io.save_object(best_params_f1, "models/xgb_best_params_f1.pkl")

    model_io.save_object(best_model_name, "models/best_model_name.pkl")
    model_io.save_object(list(X_train.columns), "models/feature_columns.pkl")
    model_io.save_object(team_df, "models/team_df.pkl")
    model_io.save_object(df_with_elo, "models/df_with_elo.pkl")

    results_df.to_csv("models/results_summary.csv", index=False)
    ranked_results.to_csv("models/results_ranked.csv", index=False)

    print("\nSaved model artifacts to models/")
    print("Saved results summary to models/results_summary.csv")
    print("Saved ranked results to models/results_ranked.csv")

    #prediction test
    predictor = MatchPredictor(
        model=best_model,
        feature_columns=list(X_train.columns),
        team_df_model=team_df,
        df_with_elo=df_with_elo
    )

    prediction_result = predictor.predict(
        home_team="arsenal",
        away_team="chelsea",
        match_date="2025-05-10"
    )

    print(f"\nSingle match prediction using best model ({best_model_name}):")
    print(prediction_result)


if __name__ == "__main__":
    main()