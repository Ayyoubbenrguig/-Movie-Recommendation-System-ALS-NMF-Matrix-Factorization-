"""
ALS complet (from scratch, numpy only) + évaluation + graphes.
Dataset: MovieLens 100k (u1.base / u1.test)

Sorties:
- ALS_matrices.pkl (P et Q)
- als_loss_curve.png
- als_confusion_matrix.png
"""

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utils: chargement données
# -----------------------------
def load_movielens_100k(path: Path) -> np.ndarray:
    """
    MovieLens 100k: userId \t movieId \t rating \t timestamp
    Retour: ndarray (n, 4)
    """
    data = np.genfromtxt(path, delimiter="\t", dtype=int)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise ValueError(f"Format inattendu: {path}")
    return data[:, :4] if data.shape[1] >= 4 else data


# -----------------------------
# ALS: entraînement
# -----------------------------
def als_train(
    train_data: np.ndarray,
    n_users: int,
    n_items: int,
    k: int = 20,
    lam: float = 0.1,
    n_iters: int = 5000,
    seed: int = 42,
):
    """
    Entraîne ALS explicite:
    - P: (n_users, k)
    - Q: (n_items, k)
    Optimise sur les ratings observés.
    """
    rng = np.random.default_rng(seed)
    P = rng.random((n_users, k))
    Q = rng.random((n_items, k))

    mse_hist = []
    rmse_hist = []

    for it in range(n_iters):
        # --- Update P (users) ---
        for u in range(n_users):
            mask = train_data[:, 0] == (u + 1)
            if np.sum(mask) == 0:
                continue
            item_ids = train_data[mask, 1] - 1  # 0-index
            ratings = train_data[mask, 2].astype(float)

            Y = Q[item_ids, :]  # (num_rated_items, k)

            # Solve: (Y^T Y + lam I) p_u = Y^T r_u
            A = (Y.T @ Y) + lam * np.eye(k)
            b = Y.T @ ratings
            P[u, :] = np.linalg.solve(A, b)

        # --- Update Q (items) ---
        for i in range(n_items):
            mask = train_data[:, 1] == (i + 1)
            if np.sum(mask) == 0:
                continue
            user_ids = train_data[mask, 0] - 1
            ratings = train_data[mask, 2].astype(float)

            X = P[user_ids, :]  # (num_users_who_rated_item, k)

            # Solve: (X^T X + lam I) q_i = X^T r_i
            A = (X.T @ X) + lam * np.eye(k)
            b = X.T @ ratings
            Q[i, :] = np.linalg.solve(A, b)

        # --- Train loss (MSE/RMSE) ---
        se = 0.0
        for row in train_data:
            u = row[0] - 1
            i = row[1] - 1
            r = float(row[2])
            pred = float(np.dot(P[u], Q[i]))
            se += (r - pred) ** 2

        mse = se / train_data.shape[0]
        rmse = float(np.sqrt(mse))
        mse_hist.append(float(mse))
        rmse_hist.append(rmse)

        print(f"Iter {it+1:02d}/{n_iters} - Train MSE={mse:.4f} RMSE={rmse:.4f}")

    return P, Q, mse_hist, rmse_hist


def save_model(P: np.ndarray, Q: np.ndarray, out_path: Path):
    with open(out_path, "wb") as f:
        pickle.dump(P, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(Q, f, pickle.HIGHEST_PROTOCOL)


# -----------------------------
# Évaluation: RMSE/MAE + confusion matrix
# -----------------------------
def round_to_half(x: float) -> float:
    return np.round(x * 2) / 2


def rating_to_class(r: float, mode: str = "int") -> int:
    """
    mode="int": classes 1..5 (5 classes) via arrondi
    mode="half": classes 0.5..5.0 (10 classes) via arrondi au 0.5
    """
    if mode == "int":
        r = float(np.clip(r, 0.5, 5.0))
        r = float(np.clip(np.round(r), 1, 5))
        return int(r) - 1  # 0..4
    elif mode == "half":
        r = float(np.clip(r, 0.5, 5.0))
        r = float(round_to_half(r))
        # 0.5 -> 0, 1.0 -> 1, ... 5.0 -> 9
        return int((r - 0.5) / 0.5)
    else:
        raise ValueError("mode must be 'int' or 'half'")


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def evaluate(
    P: np.ndarray,
    Q: np.ndarray,
    test_data: np.ndarray,
    clip: bool = True,
    lo: float = 0.5,
    hi: float = 5.0,
):
    n_users = P.shape[0]
    n_items = Q.shape[0]

    sq = 0.0
    ab = 0.0
    count = 0
    oob = 0
    skipped = 0

    y_true_cls = []
    y_pred_cls = []

    # Choix pour la matrice de confusion
    class_mode = "int"  # "int" (1..5) recommandé / "half" (0.5..5.0)

    for row in test_data:
        u = int(row[0])
        i = int(row[1])
        r_true = float(row[2])

        if not (1 <= u <= n_users and 1 <= i <= n_items):
            skipped += 1
            continue

        pred = float(np.dot(P[u - 1], Q[i - 1]))

        if pred < lo or pred > hi:
            oob += 1

        if clip:
            pred = float(np.clip(pred, lo, hi))

        err = r_true - pred
        sq += err * err
        ab += abs(err)
        count += 1

        # Confusion matrix
        y_true_cls.append(rating_to_class(r_true, mode=class_mode))
        y_pred_cls.append(rating_to_class(pred, mode=class_mode))

    mse = sq / count
    rmse = float(np.sqrt(mse))
    mae = ab / count

    if class_mode == "int":
        labels = ["1", "2", "3", "4", "5"]
        n_classes = 5
    else:
        labels = [f"{x/2:.1f}" for x in range(1, 11)]  # 0.5..5.0
        n_classes = 10

    cm = confusion_matrix(y_true_cls, y_pred_cls, n_classes=n_classes)

    return {
        "MSE": float(mse),
        "RMSE": rmse,
        "MAE": float(mae),
        "pct_oob_before_clip": 100.0 * oob / count,
        "n_test_rows": int(test_data.shape[0]),
        "n_evaluated": int(count),
        "n_skipped": int(skipped),
        "cm": cm,
        "cm_labels": labels,
    }


# -----------------------------
# Plots: loss + confusion matrix
# -----------------------------
def plot_loss(mse_hist, rmse_hist, out_path: Path):
    x = np.arange(1, len(mse_hist) + 1)

    plt.figure()
    plt.plot(x, mse_hist, label="MSE")
    plt.plot(x, rmse_hist, label="RMSE")
    plt.title("ALS - Courbe de loss pendant l'entraînement")
    plt.xlabel("Itération")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, labels, out_path: Path):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("ALS - Matrice de confusion (notes en classes)")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    # annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v > 0:
                plt.text(j, i, str(v), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
def main():
    base_dir = Path(__file__).resolve().parent

    # ✅ Adapte ces chemins si nécessaire
    train_path = (base_dir / ".." / ".." / ".." / "Data" / "Data preprocessing" / "u1.base").resolve()
    test_path = (base_dir / ".." / ".." / ".." / "Data" / "Data preprocessing" / "u1.test").resolve()

    if not train_path.exists():
        raise FileNotFoundError(f"Train introuvable: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test introuvable: {test_path}")

    train_data = load_movielens_100k(train_path)
    test_data = load_movielens_100k(test_path)

    n_users = int(train_data[:, 0].max())
    n_items = int(train_data[:, 1].max())

    # Hyperparamètres
    K = 20
    Lambda = 0.1
    n_iters = 2000  # augmente à 20 si tu veux

    print(f"Train rows={train_data.shape[0]} users={n_users} items={n_items}")
    print(f"Test  rows={test_data.shape[0]}")
    print("\n--- Entraînement ALS ---")

    P, Q, mse_hist, rmse_hist = als_train(
        train_data=train_data,
        n_users=n_users,
        n_items=n_items,
        k=K,
        lam=Lambda,
        n_iters=n_iters,
        seed=42,
    )

    # Sauvegarde modèle
    model_path = base_dir / "ALS_matrices.pkl"
    save_model(P, Q, model_path)
    print(f"\nModèle sauvegardé: {model_path}")

    # Courbe de loss
    loss_png = base_dir / "als_loss_curve.png"
    plot_loss(mse_hist, rmse_hist, loss_png)
    print(f"Courbe de loss sauvegardée: {loss_png}")

    # Évaluation
    print("\n--- Évaluation sur u1.test ---")
    metrics = evaluate(P, Q, test_data, clip=True, lo=0.5, hi=5.0)

    print(f"MSE  : {metrics['MSE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"% hors bornes avant clip: {metrics['pct_oob_before_clip']:.2f}%")
    print(f"Lignes test: {metrics['n_test_rows']} | évaluées: {metrics['n_evaluated']} | ignorées: {metrics['n_skipped']}")

    # Matrice de confusion
    cm_png = base_dir / "als_confusion_matrix.png"
    plot_confusion_matrix(metrics["cm"], metrics["cm_labels"], cm_png)
    print(f"Matrice de confusion sauvegardée: {cm_png}")


if __name__ == "__main__":
    main()
