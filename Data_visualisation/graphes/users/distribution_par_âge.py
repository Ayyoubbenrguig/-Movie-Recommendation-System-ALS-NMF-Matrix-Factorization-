# Ce code permet de créer un histogramme qui affiche la répartition des utilisateurs selon leur âge

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

dico = {}
base_dir = Path(__file__).resolve().parent
data_path = (base_dir / ".." / ".." / ".." / "Data" / "Data preprocessing" / "u.user").resolve()
with open(data_path, "r", encoding="latin-1") as file:
    for ligne in file:
        data = ligne.split("|")
        age = str(data[1])
        if age == '7':
            age = '07'
        if age in dico.keys():
            dico[age] += 1
        else:
            dico[age] = 1

# afficher l'âge avec son occurence: 
for i, j in dico.items():
    pass#print(i, j)

# tracer la fonction d'occurence et l'âge: 
X = sorted(list(dico.keys()))
occurence = []
for x in X:
    occurence.append(dico[x])

# Tracer le graphe
plt.bar(X, occurence, color='indigo', width=0.6)
plt.grid(True)
plt.title("Occurence de chaque âge")
plt.xlabel("Âge")
plt.ylabel("Occurence")
plt.text(2, 50, 'Données collectées de 2021-2023')
plt.style.use('seaborn-v0_8')
plt.show()
