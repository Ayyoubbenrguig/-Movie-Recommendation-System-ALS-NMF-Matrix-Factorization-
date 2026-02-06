import pickle
from pathlib import Path
import numpy as np

base_dir = Path(__file__).resolve().parent
default_path = base_dir / "ALS_matrices.pkl"
fallback_path = base_dir / "matrices.pkl"
model_path = default_path if default_path.exists() else fallback_path

with open(model_path, "rb") as f:
    P = pickle.load(f)
    Q = pickle.load(f)

n_users, k1 = P.shape
n_items, k2 = Q.shape
assert k1 == k2, "P et Q doivent avoir le même nombre de facteurs latents (K)."

print(f"Modèle chargé: P={P.shape}, Q={Q.shape}")

while True:
    try:
        test_user = int(input("Entrer un id d'utilisateur : "))
        test_item = int(input("Entrer un id de film : "))
    except (EOFError, KeyboardInterrupt):
        print("\nFin du test.")
        break

    if not (1 <= test_user <= n_users):
        print(f"Utilisateur invalide. Entrez un id entre 1 et {n_users}.\n")
        continue

    if not (1 <= test_item <= n_items):
        print(f"Film invalide. Entrez un id entre 1 et {n_items}.\n")
        continue

    pred = float(np.dot(P[test_user - 1], Q[test_item - 1]))
    pred_clipped = float(np.clip(pred, 0.5, 5.0))

    print(
        f"Note prédite (brute) pour user {test_user} sur film {test_item}: {pred:.2f}\n"
        f"Note prédite (clippée 0.5–5.0): {pred_clipped:.2f}\n"
    )
