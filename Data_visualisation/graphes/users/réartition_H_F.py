# Ce code permet d'afficher un graphe qui donne la répartition Hommes/Femmes dans le dataset

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

Nmfemme = 0
Nmhomme = 0
Xgenre = ["Homme", "Femme"]
mycolors = ["green", "red"]

base_dir = Path(__file__).resolve().parent
data_path = (base_dir / ".." / ".." / ".." / "Data" / "Data preprocessing" / "u.user").resolve()
with open(data_path, "r", encoding="latin-1") as file:
    for ligne in file:
        data = ligne.split("|")
        genre = str(data[2])
        if genre == 'F':
            Nmfemme += 1
        else:
            Nmhomme += 1

s = Nmfemme + Nmhomme

y = np.array([(Nmhomme/s)*100, (Nmfemme/s)*100])

mycolors = ['#4C72B0', '#C44E52']
plt.pie(y, labels=Xgenre, autopct='%1.1f%%')
plt.title('Répartition par Genre', fontsize=18)
plt.legend(title='Genre', loc='lower center')
plt.rcParams.update({'font.size': 14})
plt.text(-1.3, -1.25, 'Données collectées entre 1997 et 1998')
plt.legend(title="genre:", bbox_to_anchor=(0.9, 1), ncol=1)
plt.suptitle('Répartition des genres', fontsize=16, y=1.05)
plt.show()