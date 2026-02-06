# 🎬 Movie Recommendation System — ALS & NMF (Matrix Factorization)

Ce projet implémente un **système de recommandation de films** basé sur le **filtrage collaboratif** et la **factorisation de matrice**.  
Deux algorithmes sont développés et comparés :

- **ALS (Alternating Least Squares)**
- **NMF (Non-negative Matrix Factorization)**

Le projet couvre **l’entraînement**, **l’évaluation** et une **interface interactive de démonstration**.

---

## 📌 Objectifs du projet

- Comprendre le fonctionnement des systèmes de recommandation
- Implémenter la factorisation de matrice **from scratch**
- Comparer ALS et NMF sur le dataset **MovieLens**
- Évaluer les performances avec des métriques adaptées
- Développer une interface interactive pour la démonstration

---

## 📊 Dataset

- **Source** : MovieLens 100k (GroupLens)
- **Utilisateurs** : 943
- **Films** : 1682
- **Évaluations** : 100 000
- **Échelle des notes** : 0.5 à 5.0

Les données sont prétraitées dans le dossier `Data/Data preprocessing`.

---


## ⚙️ Méthodes utilisées

### 🔹 Factorisation de matrice

La matrice utilisateur–film \( R \) est approximée par :

\[
R \approx P \times Q^T
\]

- **P** : matrice des utilisateurs (facteurs latents)
- **Q** : matrice des films (facteurs latents)

La note prédite est donnée par :
\[
\hat{r}_{ui} = P_u \cdot Q_i
\]

---

### 🔹 ALS (Alternating Least Squares)

- Mise à jour alternée de \( P \) et \( Q \)
- Résolution par moindres carrés
- Régularisation pour éviter le sur-apprentissage
- Très efficace pour les matrices creuses

---

### 🔹 NMF (Non-negative Matrix Factorization)

- Même objectif que ALS
- Contraintes : \( P \ge 0 \) et \( Q \ge 0 \)
- Utilise des mises à jour multiplicatives
- Facteurs latents plus interprétables

---

## 🏋️ Entraînement des modèles

Les modèles sont entraînés sur le fichier `u1.base`.

- Nombre de facteurs latents : `K = 20`
- Régularisation : `λ = 0.1`
- Itérations :
  - ALS : 10–20
  - NMF : 50–100

Les matrices entraînées sont sauvegardées dans :
- `ALS_matrices.pkl`
- `NMF_matrices.pkl`

---

## 📈 Évaluation

L’évaluation est réalisée sur le fichier `u1.test`.

### 🔹 Métriques utilisées
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- Pourcentage de prédictions hors bornes

### 🔹 Visualisations
- Courbe de loss (MSE / RMSE)
- Matrice de confusion (notes transformées en classes)

---

## 🖥️ Interface de démonstration

Une interface interactive est développée avec **Streamlit**.

### Fonctionnalités :
- Choix du modèle (ALS / NMF)
- Sélection de l’utilisateur
- Affichage de l’historique des films notés
- Recommandations Top-N de films non vus
- Test de prédiction pour un couple utilisateur–film

### Lancer la démo :
```bash
streamlit run app_reco.py
