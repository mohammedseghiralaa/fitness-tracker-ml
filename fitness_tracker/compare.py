from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from load_data import load_clean, split_xy

train, test = load_clean()
X_train, y_train, g_train = split_xy(train)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "SVM": make_pipeline(StandardScaler(), SVC()),
}

cv = GroupKFold(n_splits=5)
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, groups=g_train,
                             cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"{name:14s} mean={scores.mean():.3f}  std={scores.std():.3f}")