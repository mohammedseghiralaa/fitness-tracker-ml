import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from load_data import load_clean, split_xy

train, test = load_clean()
X_train, y_train, g_train = split_xy(train)

cv = GroupKFold(n_splits=5)

searches = {
    "LogReg": (
        Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "SVM": (
        Pipeline([("sc", StandardScaler()), ("clf", SVC())]),
        {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.001, 0.0001]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [None, 20], "min_samples_leaf": [1, 3]},
    ),
}

for name, (model, grid) in searches.items():
    gs = GridSearchCV(model, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train, groups=g_train)
    print(f"{name:12s} best F1={gs.best_score_:.3f}  params={gs.best_params_}")