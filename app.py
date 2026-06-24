import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Spotify Genre Classifier", page_icon="🎵", layout="centered")

FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms",
]

@st.cache_resource
def load_artifacts():
    rf = joblib.load("model_rf.joblib")
    le = joblib.load("label_encoder.joblib")
    importance = pd.read_csv("feature_importance.csv", index_col=0)
    return rf, le, importance

@st.cache_data
def load_data():
    df = pd.read_csv("spotify_songs.csv")
    return df.drop_duplicates(subset=["track_id"]).dropna(subset=FEATURES + ["playlist_genre"])

rf, le, importance = load_artifacts()
df = load_data()

@st.cache_data
def load_rf_accuracy():
    with open("results.txt") as f:
        for line in f:
            if line.startswith("Random Forest"):
                return float(line.strip().split(":")[1]) * 100
    return None

rf_accuracy = load_rf_accuracy()

st.title("🎵 Spotify Genre Classifier")
st.caption(
    "Predicts a song's genre from its raw audio features. "
    "Trained on ~28k tracks (TidyTuesday Spotify dataset) with a Random Forest classifier "
    f"— {rf_accuracy:.1f}% accuracy across 6 genres (random baseline: ~17%)."
)

tab1, tab2 = st.tabs(["🔮 Predict a Genre", "📊 What the Model Learned"])

with tab1:
    st.subheader("Set the audio features")
    st.caption("Drag the sliders, or click a preset below to load a real track's features.")

    preset_col1, preset_col2, preset_col3 = st.columns(3)
    sample = None
    if preset_col1.button("🎲 Random EDM track"):
        sample = df[df.playlist_genre == "edm"].sample(1).iloc[0]
    if preset_col2.button("🎲 Random Rock track"):
        sample = df[df.playlist_genre == "rock"].sample(1).iloc[0]
    if preset_col3.button("🎲 Random Latin track"):
        sample = df[df.playlist_genre == "latin"].sample(1).iloc[0]

    def val(col, default):
        return float(sample[col]) if sample is not None else default

    c1, c2 = st.columns(2)
    with c1:
        danceability = st.slider("Danceability", 0.0, 1.0, val("danceability", 0.65))
        energy = st.slider("Energy", 0.0, 1.0, val("energy", 0.70))
        valence = st.slider("Valence (musical positivity)", 0.0, 1.0, val("valence", 0.50))
        acousticness = st.slider("Acousticness", 0.0, 1.0, val("acousticness", 0.10))
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, val("instrumentalness", 0.01))
        liveness = st.slider("Liveness", 0.0, 1.0, val("liveness", 0.15))
    with c2:
        speechiness = st.slider("Speechiness", 0.0, 1.0, val("speechiness", 0.08))
        loudness = st.slider("Loudness (dB)", -30.0, 2.0, val("loudness", -6.0))
        tempo = st.slider("Tempo (BPM)", 50.0, 220.0, val("tempo", 120.0))
        duration_ms = st.slider("Duration (sec)", 60, 480, int(val("duration_ms", 200000) / 1000)) * 1000
        key = st.selectbox("Key", list(range(12)), index=int(val("key", 0)))
        mode = st.selectbox("Mode", [0, 1], index=int(val("mode", 1)), format_func=lambda x: "Minor" if x == 0 else "Major")

    if sample is not None:
        st.info(f"Loaded: **{sample['track_name']}** by {sample['track_artist']} (actual genre: **{sample['playlist_genre']}**)")

    X = pd.DataFrame([{
        "danceability": danceability, "energy": energy, "key": key, "loudness": loudness,
        "mode": mode, "speechiness": speechiness, "acousticness": acousticness,
        "instrumentalness": instrumentalness, "liveness": liveness, "valence": valence,
        "tempo": tempo, "duration_ms": duration_ms,
    }])[FEATURES]

    proba = rf.predict_proba(X)[0]
    pred_idx = np.argmax(proba)
    pred_genre = le.inverse_transform([pred_idx])[0]

    st.subheader(f"Predicted genre: **{pred_genre.upper()}**")
    proba_df = pd.DataFrame({"genre": le.classes_, "probability": proba}).sort_values("probability", ascending=False)
    st.bar_chart(proba_df.set_index("genre"))

with tab2:
    st.subheader("Which features matter most")
    st.bar_chart(importance)
    st.caption("Random Forest feature importance — speechiness, tempo, and danceability carry the most signal.")

    st.subheader("Average feature values by genre")
    avg_by_genre = df.groupby("playlist_genre")[FEATURES].mean()
    feature_to_plot = st.selectbox("Compare a feature across genres", FEATURES, index=FEATURES.index("danceability"))
    st.bar_chart(avg_by_genre[feature_to_plot])

st.divider()
st.caption(
    "Built as a from-scratch rebuild of a coursework project (originally trained in R) — "
    "this version was built end-to-end with an AI coding agent (data pull → training script → this app) "
    "to ship a real, working, live version instead of a dead link."
)
