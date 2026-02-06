from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# changer la limite d'affichage de pandas à None
pd.set_option('display.max_rows', None)

# charger les donnees a partir d'un fichier (liste des utilisateurs)
base_dir = Path(__file__).resolve().parent
data_path = (base_dir / ".." / ".." / ".." / "Data" / "Data preprocessing" / "u.user").resolve()
data = pd.read_csv(
    data_path,
    sep="|",
    header=None,
    names=["user_id", "age", "gender", "occupation", "zip_code"],
)

# grouper les donnees par catégorie et compter le nombre d'utilisateurs
ratings_count = data.groupby("occupation")["user_id"].count()



# Sélectionner les 30 premiers utilisateurs
ratings_count_top30 = ratings_count.head(30)

# Afficher les résultats
max_ratings = ratings_count.max()
max_users = ratings_count[ratings_count == max_ratings].index.tolist()
min_ratings = ratings_count.min()
min_users = ratings_count[ratings_count == min_ratings].index.tolist()


print(f"La/Les Films qui ont le nombre maximal de votes ({max_ratings}) sont les utilisateurs : {max_users}.")
print(f"La/Les Films qui ont le nombre minimal de votes ({min_ratings}) sont les utilisateurs : {min_users}.")

# mettre le résultat dans un fichier count.txt
with open("count.txt", "w", encoding='utf-8') as file:
    file.write(str(ratings_count))

# Créer un graphique à barres pour les 30 premiers utilisateurs
ratings_count_top30.plot(kind='bar')

# Ajouter des étiquettes et un titre
plt.xlabel("Catégorie sociale")
plt.ylabel("Nombre de personnes")
plt.title("Nombre de personnes par catégorie")

# Afficher le graphique
plt.show()