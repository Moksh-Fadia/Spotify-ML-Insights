import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
st.set_page_config(page_title="Spotify ML Project", layout="wide")
st.title("Spotify ML Project Dashboard")

# --- LOAD DATA ---
data_folder = "/Users/mokshfadia/Desktop/Data analysis projects/Spotify project" 
spotify_df = pd.read_csv("spotify.csv")

playlist_files = {
    "High-Energy Electronic": "high-energy_electronic.csv",
    "Chill Indie": "chill_indie.csv",
    "Slow Sad Acoustic": "slow_sad_acoustic.csv",
    "Danceable Pop Vibes": "danceable_pop_vibes.csv"
}

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")

section = st.sidebar.radio("Choose Feature", [
    "Mood Based Clustering",
    "Song Recommender",
    "Hit/Flop Prediction",
    "Genre Classification"
])

# --- SHARED SETUP ---
def scale_features(df, features):
    scaler = StandardScaler()
    return scaler.fit_transform(df[features])


# --- SECTION 1: GENRE CLASSIFICATION ---
if section == "Genre Classification":
    st.header("🎵 Genre Classification")
    genre_df = spotify_df.copy()
    genre_df.dropna(inplace=True)

    X = genre_df.drop(columns=['track_id', 'artists', 'album_name', 'track_name', 'track_genre'])
    y = LabelEncoder().fit_transform(genre_df['track_genre'])
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    xgb = XGBClassifier(
        objective='multi:softmax',
        num_class=len(np.unique(y)),
        eval_metric='mlogloss',
        random_state=42
    )

    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    st.subheader("Classification Report (Train/Test Split)")
    st.code(classification_report(y_test, y_pred))

    scores = cross_val_score(xgb, X_scaled, y, cv=5, scoring='f1_weighted')
    st.subheader("K-Fold Cross Validation (F1 Weighted)")
    st.write("Scores:", scores)
    st.write("Average F1 Score:", np.round(scores.mean(), 4))
    
    st.markdown("""
     **Note:** The classification report above is based on a single 80/20 train-test split.
     The K-Fold Cross Validation gives a more stable and generalized estimate of model performance.
    """)


# --- SECTION: Hit/Flop Prediction ---
if section == "Hit/Flop Prediction":
    st.header("🔥 Hit or Flop Classifier")

    hit_df = spotify_df.copy()
    hit_df['is_hit'] = (hit_df['popularity'] >= 60).astype(int)

    X = hit_df.drop(columns=['track_id', 'artists', 'album_name', 'track_name', 'track_genre', 'popularity', 'is_hit'])
    y = hit_df['is_hit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    st.subheader("🎯 Guess a Song's Hit Potential")
    song_input = st.text_input("Enter Song Name")

    if song_input:
        matches = hit_df[hit_df['track_name'].str.lower() == song_input.lower()]

        if matches.empty:
            st.error("Song not found in dataset.")
        else:
            if len(matches) > 1:
                st.write("Multiple matches found. Choose the correct artist.")
                artist_selected = st.selectbox("Select Artist", matches['artists'].unique())
                song_row = matches[matches['artists'] == artist_selected].iloc[0]
            else:
                song_row = matches.iloc[0]

            song_features = song_row[X.columns].values.reshape(1, -1)
            song_scaled = scaler.transform(song_features)
            pred = model.predict(song_scaled)[0]
            prob = model.predict_proba(song_scaled)[0][1]

            st.markdown(f"**{song_row['track_name']}** by *{song_row['artists']}*")
            st.success("Predicted: **Hit**" if pred == 1 else "Predicted: **Flop**")
            st.markdown(f"**Hit Probability:** `{prob:.2f}`")

    st.markdown("---")
    st.subheader("📊 Model Evaluation (Train/Test Split)")
    st.text("Classification Report on Unseen Test Data:")
    y_pred = model.predict(X_test_scaled)
    st.code(classification_report(y_test, y_pred, target_names=["Flop", "Hit"]))

    st.markdown(" **Note:** The model is evaluated on a held-out test set (20%).")


# --- SECTION 3: MOOD CLUSTERING ---
elif section == "Mood Based Clustering":
    st.header("🎭 Mood-Based Clustering")
    cluster_df = spotify_df.copy()
    mood_features = ['valence', 'energy', 'danceability', 'tempo', 'acousticness']
    X_scaled = scale_features(cluster_df, mood_features)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_df['mood_cluster'] = kmeans.fit_predict(X_scaled)

    mood_names = {
        0: "High-Energy Electronic",
        1: "Chill Indie",
        2: "Slow Sad Acoustic",
        3: "Danceable Pop Vibes"
    }
    cluster_df['playlist_name'] = cluster_df['mood_cluster'].map(mood_names)

    selected = st.selectbox("Choose Playlist", list(playlist_files.keys()))
    st.dataframe(pd.read_csv(playlist_files[selected])[['track_name', 'artists']])


# --- SECTION 4: RECOMMENDER ---
elif section == "Song Recommender":
    st.header("🎯 Song Recommender")
    reco_df = spotify_df.copy()

    X_scaled = scale_features(reco_df, ['valence', 'energy', 'danceability', 'tempo', 'acousticness'])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    reco_df['mood_cluster'] = kmeans.fit_predict(X_scaled)

    mood_names = {
        0: "High-Energy Electronic",
        1: "Chill Indie",
        2: "Slow Sad Acoustic",
        3: "Danceable Pop Vibes"
    }
    reco_df['playlist_name'] = reco_df['mood_cluster'].map(mood_names)

    features = ['valence', 'energy', 'danceability', 'tempo', 'acousticness']
    reco_scaled = StandardScaler().fit_transform(reco_df[features])
    similarity_matrix = cosine_similarity(reco_scaled)

    name = st.text_input("Enter Song Name")
    artist = st.text_input("Enter Artist Name")

    if st.button("Recommend Similar Songs"):
        idx = reco_df[(reco_df['track_name'].str.lower() == name.lower()) & 
                      (reco_df['artists'].str.lower().str.contains(artist.lower()))].index

        if len(idx) == 0:
            st.error("Song not found. Try checking spelling.")
        else:
            idx = idx[0]
            scores = list(enumerate(similarity_matrix[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]
            shown = set()
            count = 0
            st.subheader(f"Because you liked: {reco_df.loc[idx, 'track_name']} by {reco_df.loc[idx, 'artists']}")
            for i, _ in scores:
                title = reco_df.loc[i, 'track_name']
                if title in shown:
                    continue
                shown.add(title)
                artist_name = reco_df.loc[i, 'artists']
                playlist = reco_df.loc[i, 'playlist_name']
                st.markdown(f"→ **{title}** — {artist_name} _({playlist})_")
                count += 1
                if count == 5:
                    break
