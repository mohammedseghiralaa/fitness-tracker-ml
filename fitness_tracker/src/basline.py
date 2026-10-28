from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from fitness_tracker.src.load_data import load_clean, split_xy

train, test = load_clean()
X_train, y_train, g_train = split_xy(train)
X_test, y_test, _ = split_xy(test)

model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)

# Cross-validation on train, splitting by subject (new people in each fold)
cv = GroupKFold(n_splits=5)
scores = cross_val_score(model, X_train, y_train, groups=g_train, cv=cv)
print("CV accuracy per fold:", scores.round(3))
print("CV mean:", scores.mean().round(3))

# Final check on the held-out test set
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))