import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fitness_tracker.load_data import load_clean, split_xy


def main():
    train, _ = load_clean()
    X, y, g = split_xy(train)
    cv = GroupKFold(n_splits=5)

    searches = {
        "LogReg": (
            Pipeline([("sc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000))]),
            {
                "clf__C": [0.01, 0.03, 0.1, 0.3, 1],
                "clf__penalty": ["l2"],
            },
        ),
        "SVM": (
            Pipeline([("sc", StandardScaler()), ("clf", SVC())]),
            {
                "clf__C": [3, 10, 30, 100],
                "clf__gamma": [0.0003, 0.001, 0.003],
                "clf__kernel": ["rbf"],
            },
        ),
    }

    rows, best_models = [], {}
    for name, (model, grid) in searches.items():
        gs = GridSearchCV(model, grid, cv=cv, scoring="f1_macro",
                          n_jobs=-1, return_train_score=True)
        gs.fit(X, y, groups=g)
        print(f"{name}: F1={gs.best_score_:.4f}  {gs.best_params_}")
        res = pd.DataFrame(gs.cv_results_)
        res["model"] = name
        rows.append(res[["model", "params", "mean_test_score",
                         "std_test_score", "mean_train_score"]])
        best_models[name] = gs.best_estimator_

    pd.concat(rows).sort_values("mean_test_score", ascending=False).to_csv(
        "reports/tuning_results.csv", index=False
    )
    joblib.dump(best_models, "models/tuned_models.joblib")
    print("Saved reports/tuning_results.csv")


if __name__ == "__main__":
    main()