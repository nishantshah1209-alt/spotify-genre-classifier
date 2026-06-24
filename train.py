"""
train.py

Trains classifiers to predict a song's genre from its Spotify audio features.
Dataset: TidyTuesday Spotify Songs (CC0), ~32.8k tracks across 6 genres.

Models: Logistic Regression, Decision Tree, Random Forest
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms",
]
TARGET = "playlist_genre"

def load_data(path="spotify_songs.csv"):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["track_id"])
    df = df.dropna(subset=FEATURES + [TARGET])
    return df

def main():
    df = load_data()
    print(f"Loaded {len(df)} tracks across {df[TARGET].nunique()} genres")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    pred = log_reg.predict(X_test_scaled)
    results["Logistic Regression"] = accuracy_score(y_test, pred)

    # Decision Tree
    tree = DecisionTreeClassifier(max_depth=10, random_state=42)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    results["Decision Tree"] = accuracy_score(y_test, pred)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=80, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    results["Random Forest"] = accuracy_score(y_test, pred)

    print("\n=== Model Accuracy on Held-Out Test Set ===")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:22s} {acc:.3f}")

    print("\n=== Random Forest Classification Report ===")
    print(classification_report(y_test, rf.predict(X_test), target_names=le.classes_))

    # Feature importance from the best model (Random Forest)
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n=== Top Features (Random Forest) ===")
    print(importances)

    # Save artifacts for the app
    joblib.dump(rf, "model_rf.joblib", compress=3)
    joblib.dump(log_reg, "model_logreg.joblib", compress=3)
    joblib.dump(tree, "model_tree.joblib", compress=3)
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(le, "label_encoder.joblib")
    importances.to_csv("feature_importance.csv")

    with open("results.txt", "w") as f:
        f.write("Model Accuracy on Held-Out Test Set\n")
        for name, acc in sorted(results.items(), key=lambda x: -x[1]):
            f.write(f"{name}: {acc:.4f}\n")

    print("\nSaved model_rf.joblib, model_logreg.joblib, model_tree.joblib, scaler.joblib, label_encoder.joblib")

if __name__ == "__main__":
    main()
