"""
Évaluation d'un modèle ALS (Matrix Factorization) entraîné "from scratch"
à partir des matrices P (users x K) et Q (items x K) sauvegardées en pickle.

Compatible MovieLens 100k: u1.base / u1.test (séparateur tab).
Calcule: MSE, RMSE, MAE + stats sur clipping.

Ajuste juste les chemins si besoin.
"""

from pathlib import Path
import pickle
import numpy as np


def load_matrices(model_path: Path):
    with open(model_path, "rb") as f:
        P = pickle.load(f)
        Q = pickle.load(f)
    if P.ndim != 2 or Q.ndim != 2:
        raise ValueError("P et Q doivent être des matrices 2D.")
    if P.shape[1] != Q.shape[1]:
        raise ValueError(f"Incohérence K: P={P.shape}, Q={Q.shape}")
    return P, Q


def load_movielens_split(path: Path) -> np.ndarray:
    """
    Charge un fichier MovieLens 100k du type u1.base / u1.test:
    userId \t movieId \t rating \t timestamp
    Retour: np.ndarray shape (n, 4)
    """
    data = np.genfromtxt(path, delimiter="\t", dtype=int)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Format inattendu dans {path} (colonnes < 3).")
    return data[:, :4] if data.shape[1] >= 4 else data


def predict(P: np.ndarray, Q: np.ndarray, user_id: int, item_id: int) -> float:
    """Prédit la note (brute) pour user_id et item_id (ids commencent à 1)."""
    u = user_id - 1
    i = item_id - 1
    return float(np.dot(P[u], Q[i]))


def evaluate(P: np.ndarray, Q: np.ndarray, test_data: np.ndarray,
             clip: bool = True, min_rating: float = 0.5, max_rating: float = 5.0):
    """
    Évalue sur test_data (colonnes: userId, movieId, rating, timestamp).
    Retourne un dict de métriques.
    """
    n_users, _ = P.shape
    n_items, _ = Q.shape

    # On va accumuler erreurs uniquement sur les points évaluables
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    count = 0

    out_of_bounds = 0
    skipped = 0

    for row in test_data:
        user_id = int(row[0])
        item_id = int(row[1])
        true_rating = float(row[2])

        # Vérifier que les ids existent dans P et Q
        if not (1 <= user_id <= n_users) or not (1 <= item_id <= n_items):
            skipped += 1
            continue

        pred = predict(P, Q, user_id, item_id)

        # Stat sur hors bornes avant clipping
        if pred < min_rating or pred > max_rating:
            out_of_bounds += 1

        if clip:
            pred = float(np.clip(pred, min_rating, max_rating))

        err = true_rating - pred
        sq_err_sum += err * err
        abs_err_sum += abs(err)
        count += 1

    if count == 0:
        raise RuntimeError("Aucune ligne test évaluée (ids hors limites ? mauvais split ?).")

    mse = sq_err_sum / count
    rmse = float(np.sqrt(mse))
    mae = abs_err_sum / count

    return {
        "n_test_rows": int(test_data.shape[0]),
        "n_evaluated": int(count),
        "n_skipped_out_of_range_ids": int(skipped),
        "pct_out_of_bounds_before_clipping": 100.0 * out_of_bounds / count,
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "clip_enabled": bool(clip),
        "clip_range": (min_rating, max_rating),
    }


def main():
    base_dir = Path(__file__).resolve().parent

    # 1) Chemin vers le modèle (matrices picklées)
    default_model = base_dir / "ALS_matrices.pkl"
    fallback_model = base_dir / "matrices.pkl"
    model_path = default_model if default_model.exists() else fallback_model

    if not model_path.exists():
        raise FileNotFoundError(
            f"Aucun modèle trouvé: {default_model} ni {fallback_model} n'existent."
        )

    # 2) Chemins vers les données test
    # Mets ici le chemin exact vers u1.test
    test_path = (base_dir / ".." / ".." / ".." / "Data" / "Data preprocessing" / "u1.test").resolve()

    if not test_path.exists():
        raise FileNotFoundError(f"Fichier test introuvable: {test_path}")

    print(f"Chargement du modèle: {model_path}")
    P, Q = load_matrices(model_path)
    print(f"P shape = {P.shape}, Q shape = {Q.shape}")

    print(f"Chargement du test: {test_path}")
    test_data = load_movielens_split(test_path)
    print(f"Test rows = {test_data.shape[0]}")

    # 3) Évaluation (avec clipping recommandé)
    metrics = evaluate(P, Q, test_data, clip=True, min_rating=0.5, max_rating=5.0)

    print("\n--- Résultats d'évaluation (Test) ---")
    print(f"Lignes test total          : {metrics['n_test_rows']}")
    print(f"Lignes évaluées            : {metrics['n_evaluated']}")
    print(f"Lignes ignorées (ids hors) : {metrics['n_skipped_out_of_range_ids']}")
    print(f"Hors bornes avant clip (%) : {metrics['pct_out_of_bounds_before_clipping']:.2f}%")
    print(f"MSE                        : {metrics['MSE']:.4f}")
    print(f"RMSE                       : {metrics['RMSE']:.4f}")
    print(f"MAE                        : {metrics['MAE']:.4f}")
    print(f"Clipping                   : {metrics['clip_enabled']} {metrics['clip_range']}")

    # Option: afficher quelques exemples de prédiction
    print("\nExemples (5 premières lignes du test):")
    for row in test_data[:5]:
        u, i, r = int(row[0]), int(row[1]), float(row[2])
        if 1 <= u <= P.shape[0] and 1 <= i <= Q.shape[0]:
            p = predict(P, Q, u, i)
            pc = float(np.clip(p, 0.5, 5.0))
            print(f"user={u:3d} item={i:4d} true={r:.1f} pred={p:.2f} pred_clip={pc:.2f}")
        else:
            print(f"user={u:3d} item={i:4d} true={r:.1f} -> SKIPPED (id hors range)")


if __name__ == "__main__":
    main()
