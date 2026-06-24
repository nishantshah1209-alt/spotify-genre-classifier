# Spotify Genre Classifier

Predicts a song's genre from its raw audio features (danceability, energy, tempo, valence, etc.) using models trained on ~28k real Spotify tracks.

**[Live demo →](#)** *(add your deployed Streamlit URL here)*

## Background

This is a rebuild of a project I originally did in coursework — training logistic regression, decision tree, and other models in R to predict song genre from historical Spotify data. My university account (and the original files) were deactivated after graduation, so this is a from-scratch Python rebuild: same idea, real data, but now a live, working app instead of a dead link.

## What it does

- Trains three classifiers (Logistic Regression, Decision Tree, Random Forest) to predict genre across 6 categories: EDM, Latin, Pop, R&B, Rap, Rock.
- Random Forest performs best at **55.5% accuracy** (random baseline for 6 classes: ~17%).
- An interactive app lets you set audio features with sliders (or load a real track's actual values) and see the model's live genre prediction and confidence breakdown.
- A second tab shows which audio features matter most to the model, and how features differ by genre on average.

## Data

[TidyTuesday Spotify Songs dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21) (CC0) — ~32.8k tracks pulled from Spotify's API, including playlist genre/subgenre labels and Spotify's own audio feature scores (danceability, energy, valence, etc.).

## Run it locally

```bash
pip install -r requirements.txt
python train.py      # retrains models from scratch (takes ~30 seconds)
streamlit run app.py
```

## Deploy it (free, ~5 minutes)

1. Push this repo to your own GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click "New app," point it at this repo, set the main file to `app.py`, deploy.
4. You'll get a public `*.streamlit.app` URL.

## Files

- `train.py` — loads data, trains all three models, saves them and evaluation metrics
- `app.py` — the Streamlit app (prediction UI + feature analysis)
- `spotify_songs.csv` — the dataset
- `model_*.joblib`, `scaler.joblib`, `label_encoder.joblib` — trained model artifacts
- `results.txt`, `feature_importance.csv` — evaluation output from training

## Stack

Python, pandas, scikit-learn, Streamlit
