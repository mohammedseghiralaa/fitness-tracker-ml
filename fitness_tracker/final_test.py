import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fitness_tracker.load_data import load_clean, split_xy
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def main():
    train, test = load_clean()
    X_tr, y_tr, _ = split_xy(train)
    X_te, y_te, _ = split_xy(test)

    model = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=0.3, max_iter=5000)),
    ])
    model.fit(X_tr, y_tr)        # التدريب على train فقط
    pred = model.predict(X_te)   # الاختبار على test مرة واحدة


    ConfusionMatrixDisplay.from_predictions(
    y_te, pred, xticks_rotation=45, cmap="Blues"
     )
    plt.tight_layout()
    plt.savefig("reports/figures/09_confusion_matrix.png", dpi=150)

    print("Test macro-F1:", round(f1_score(y_te, pred, average="macro"), 4))
    print(classification_report(y_te, pred))
    print(confusion_matrix(y_te, pred))

    joblib.dump(model, "models/logreg_pipeline.joblib")
    print("Saved models/logreg_pipeline.joblib")


if __name__ == "__main__":
    main()