# Ce code permet de nettoyer la data des caractères non reconnus concernant le nombre d’occurrences
# des villes à partir de leurs zip codes,
# du fichier u.item, il permet aussi de trier la liste par ordre alphabétique des villes

from pathlib import Path

import pandas as pd  

base_dir = Path(__file__).resolve().parent
map_dir = (base_dir / ".." / ".." / "Data_visualisation" / "graphes" / "map_visu_users").resolve()
input_path = map_dir / "data_map.csv"
output_path = map_dir / "data_map_clean.csv"

l = []
with open(input_path, "r", encoding="utf-8") as file, open(output_path, "w", encoding="utf-8") as file2:
    for ligne in file:
        count, ville = ligne.split("\t")
        #print(count, ville[:-2])
        if [count, ville[:-1]] not in l:
            l.append([count, ville[:-1]])
            l = sorted(l, key=lambda x: x[1])
    l = l[1:-1]
    for ville in l:
        ville = "\t".join(ville) + "\n"
        print(ville)
        file2.write(ville)      