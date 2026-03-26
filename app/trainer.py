import numpy as np
import lightgbm as lgb

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


class ModelTrainer:

    def compute_sample_weights(self, y_train):
        classes = np.unique(y_train)

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )

        class_weight_dict = dict(zip(classes, class_weights))
        sample_weights = y_train.map(class_weight_dict).astype(float)

        return sample_weights, class_weight_dict

    def train_random_forest(self, X_train, y_train):
        model = RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def train_xgb(self, X_train, y_train, X_valid, y_valid, sample_weights=None):
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=4000,
            learning_rate=0.02,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
            eval_metric="mlogloss",
            early_stopping_rounds=100
        )

        if sample_weights is not None:
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                eval_set=[(X_valid, y_valid)],
                verbose=100
            )
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=100
            )

        return model

    def tune_xgb_search(self, X_train, y_train, scoring="accuracy", n_splits=5, n_iter=100):
        tscv = TimeSeriesSplit(n_splits=n_splits)

        base_model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1
        )

        param_dist = {
            "n_estimators": [300, 500, 800, 1200, 1800, 2500, 3500],
            "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
            "max_depth": [3, 4, 5, 6, 8, 10],
            "min_child_weight": [1, 2, 3, 5, 7, 10],
            "subsample": [0.65, 0.75, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.65, 0.75, 0.8, 0.9, 1.0],
            "gamma": [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
            "reg_alpha": [0, 0.001, 0.01, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
        }

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=tscv,
            verbose=2,
            random_state=42,
            n_jobs=-1,
            refit=True
        )

        search.fit(X_train, y_train, sample_weight=sample_weights)

        return search.best_params_, search.best_score_

    def fit_best_xgb_with_early_stopping(
        self,
        X_train,
        y_train,
        X_valid,
        y_valid,
        best_params,
        sample_weights=None
    ):
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            tree_method="hist",
            eval_metric="mlogloss",
            early_stopping_rounds=100,
            **best_params
        )

        if sample_weights is not None:
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                eval_set=[(X_valid, y_valid)],
                verbose=100
            )
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=100
            )

        return model

    def train_lgbm(self, X_train, y_train, X_valid, y_valid, sample_weights=None):
        model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=5000,
            learning_rate=0.02,
            max_depth=-1,
            num_leaves=63,
            min_child_samples=10,
            reg_alpha=0.0,
            reg_lambda=0.5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(100)
            ]
        )

        return model

    def train_catboost(self, X_train, y_train, X_valid, y_valid, sample_weights=None):
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="MultiClass",
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5,
            random_seed=42,
            verbose=200
        )

        if sample_weights is not None:
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
                eval_set=(X_valid, y_valid),
                use_best_model=True
            )
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
                use_best_model=True
            )

        return model