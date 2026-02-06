# Ce code utilise le fichier u.item pour récupérer le nom des films et remplace l'id des films par leurs noms
# dans un fichier uf.data, ceci est pour que les statistiques affiches les noms des films et non pas leurs ids

# ouverture du fichier pour les films pour avoir les noms
from pathlib import Path

base_dir = Path(__file__).resolve().parent
data_dir = (base_dir / ".." / ".." / ".." / "Data" / "Data preprocessing").resolve()
films_path = data_dir / "u.item"
ratings_path = data_dir / "u.data"
output_path = base_dir / "uf.data"

dico_films = {}
with open(films_path, "r", encoding="latin-1") as fichier_films:
    for ligne in fichier_films:
        ligne_donnée = ligne.split("|")
        dico_films[ligne_donnée[0]] = ligne_donnée[1]

# ouverture du fichier pour les notations
with open(ratings_path, "r", encoding="latin-1") as fichier_notations:
    with open(output_path, "w", encoding="utf-8") as fichier_nouveau:
        for ligne in fichier_notations:
            ligne_donnée = ligne.split("\t")
            ligne_donnée[1] = str(dico_films[ligne_donnée[1]]).strip()
            fichier_nouveau.write("\t".join(ligne_donnée))
