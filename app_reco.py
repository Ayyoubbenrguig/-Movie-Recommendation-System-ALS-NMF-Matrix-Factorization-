import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# --------------------
# PATHS (à adapter)
# --------------------
BASE_DIR = Path(__file__).resolve().parent

# modèles (pickles)
ALS_PATH = BASE_DIR / "ALS_matrices.pkl"
NMF_PATH = BASE_DIR / "NMF_matrices.pkl"

# données MovieLens 100k (titres + ratings)
DATA_DIR = (BASE_DIR / "Data" / "Data preprocessing").resolve()
RATINGS_PATH = DATA_DIR / "u1.base"   # ou u.data si tu préfères
ITEMS_PATH = DATA_DIR / "u.item"      # titres des films


# --------------------
# Chargements
# --------------------
@st.cache_data
def load_items(u_item_path: Path) -> pd.DataFrame:
    """
    MovieLens 100k u.item est séparé par | et encodage latin-1.
    Colonnes: movie_id | title | release_date | ... (beaucoup de colonnes)
    """
    df = pd.read_csv(
        u_item_path,
        sep="|",
        header=None,
        encoding="latin-1",
        usecols=[0, 1],
        names=["movieId", "title"],
    )
    return df


@st.cache_data
def load_ratings(u_path: Path) -> pd.DataFrame:
    """
    u1.base: userId \t movieId \t rating \t timestamp
    """
    df = pd.read_csv(
        u_path,
        sep="\t",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )
    return df


@st.cache_resource
def load_model(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        P = pickle.load(f)
        Q = pickle.load(f)
    return P, Q


def predict_score(P: np.ndarray, Q: np.ndarray, user_id: int, movie_id: int) -> float:
    return float(np.dot(P[user_id - 1], Q[movie_id - 1]))


def recommend_top_n(
    P: np.ndarray,
    Q: np.ndarray,
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    user_id: int,
    n: int = 10,
    clip: bool = True,
    lo: float = 0.5,
    hi: float = 5.0,
) -> pd.DataFrame:
    """
    Recommande Top-N films NON notés par user_id.
    """
    n_users, k1 = P.shape
    n_items, k2 = Q.shape
    assert k1 == k2

    # films déjà notés
    seen = set(ratings_df.loc[ratings_df["userId"] == user_id, "movieId"].astype(int).tolist())

    # scores pour tous les films
    user_vec = P[user_id - 1]  # (K,)
    scores = Q @ user_vec      # (n_items,) car Q: (items,K)

    if clip:
        scores = np.clip(scores, lo, hi)

    # enlever les films déjà vus
    all_movie_ids = np.arange(1, n_items + 1)
    mask_unseen = np.array([mid not in seen for mid in all_movie_ids], dtype=bool)

    unseen_ids = all_movie_ids[mask_unseen]
    unseen_scores = scores[mask_unseen]

    # top n
    top_idx = np.argsort(-unseen_scores)[:n]
    rec_movie_ids = unseen_ids[top_idx]
    rec_scores = unseen_scores[top_idx]

    rec_df = pd.DataFrame({"movieId": rec_movie_ids, "predicted_rating": rec_scores})
    rec_df = rec_df.merge(items_df, on="movieId", how="left")
    rec_df = rec_df[["movieId", "title", "predicted_rating"]]
    return rec_df


def user_history(
    ratings_df: pd.DataFrame,
    items_df: pd.DataFrame,
    user_id: int,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Affiche les top films déjà notés par l'utilisateur (les mieux notés).
    """
    hist = ratings_df[ratings_df["userId"] == user_id].copy()
    hist = hist.merge(items_df, on="movieId", how="left")
    hist = hist.sort_values(["rating", "timestamp"], ascending=[False, False]).head(top_k)
    return hist[["movieId", "title", "rating"]]


# --------------------
# UI Streamlit
# --------------------
st.set_page_config(page_title="Movie Recommender (ALS/NMF)", layout="wide")

st.title("🎬 Movie Recommender System — ALS / NMF (Matrix Factorization)")

# vérifs fichiers
missing = []
for p in [ITEMS_PATH, RATINGS_PATH]:
    if not p.exists():
        missing.append(str(p))
if missing:
    st.error("Fichiers de données manquants:\n" + "\n".join(missing))
    st.stop()

items_df = load_items(ITEMS_PATH)
ratings_df = load_ratings(RATINGS_PATH)

# Sidebar
st.sidebar.header("⚙️ Paramètres")
model_name = st.sidebar.selectbox("Choisir le modèle", ["ALS", "NMF"])

if model_name == "ALS":
    if not ALS_PATH.exists():
        st.sidebar.error(f"Modèle introuvable: {ALS_PATH}")
        st.stop()
    P, Q = load_model(ALS_PATH)
else:
    if not NMF_PATH.exists():
        st.sidebar.error(f"Modèle introuvable: {NMF_PATH}")
        st.stop()
    P, Q = load_model(NMF_PATH)

n_users = P.shape[0]
n_items = Q.shape[0]

user_id = st.sidebar.number_input("User ID", min_value=1, max_value=int(n_users), value=1, step=1)
top_n = st.sidebar.slider("Nombre de recommandations (Top-N)", min_value=5, max_value=30, value=10, step=1)

clip = st.sidebar.checkbox("Clipper les notes prédites dans [0.5, 5.0]", value=True)

st.sidebar.caption(f"Users: {n_users} | Items: {n_items} | K: {P.shape[1]}")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⭐ Historique utilisateur (films déjà notés)")
    hist_df = user_history(ratings_df, items_df, int(user_id), top_k=15)
    st.dataframe(hist_df, use_container_width=True)

with col2:
    st.subheader("✅ Recommandations (films non vus)")
    rec_df = recommend_top_n(
        P, Q, ratings_df, items_df,
        user_id=int(user_id),
        n=int(top_n),
        clip=clip
    )
    st.dataframe(rec_df, use_container_width=True)

st.markdown("---")
st.subheader("🔎 Tester une prédiction user/film")
movie_id = st.number_input("Movie ID", min_value=1, max_value=int(n_items), value=1, step=1)
pred = predict_score(P, Q, int(user_id), int(movie_id))
pred_clip = float(np.clip(pred, 0.5, 5.0))
title = items_df.loc[items_df["movieId"] == int(movie_id), "title"].head(1).values
title = title[0] if len(title) else "(titre introuvable)"

st.write(f"Film: **{title}**")
st.write(f"Note prédite (brute): **{pred:.2f}**")
st.write(f"Note prédite (clippée): **{pred_clip:.2f}**")
