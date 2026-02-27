# TARDIS 

## Présentation

TARDIS est un projet d'analyse de données appliqué aux retards des trains SNCF.
L'objectif est de nettoyer et explorer un jeu de données historique, d'entraîner un modèle de machine learning pour prédire la durée des retards, et de rendre ces
insights accessibles via un dashboard Streamlit interactif.

## Fichiers du projet

Fichier : 
    -`tardis_eda.ipynb` : Nettoyage, exploration et feature engineering
    -`tardis_model.ipynb` : Entraînement et évaluation du modèle 
    -`tardis_dashboard.py`: Dashboard interactif Streamlit
    -`cleaned_dataset.csv`: Dataset nettoyé produit par le notebook EDA 
    -`model.joblib`: Modèle entraîné sauvegardé 
    -`requirements.txt`: Dépendances du projet 


## Utilisation

### 1. Exploration & Nettoyage des données

Ouvrir et exécuter `tardis_eda.ipynb` dans Jupyter :
```bash
jupyter notebook tardis_eda.ipynb
```

Ce notebook produit le fichier `cleaned_dataset.csv`.

### 2. Entraînement du modèle

Ouvrir et exécuter `tardis_model.ipynb` :
```bash
jupyter notebook tardis_model.ipynb
```

Ce notebook produit le fichier `model.pkl`.

### 3. Lancer le dashboard
```bash
streamlit run tardis_dashboard.py
```

Le dashboard est accessible sur `http://localhost:8501`.

---

## Dashboard — Fonctionnalités

- **Visualisation des distributions** de retards
- **Statistiques clés** : retard moyen, taux de ponctualité, nombre de trajets
- **Interface de prédiction** : saisir les paramètres d'un trajet et obtenir un retard estimé
- **Filtres interactifs** par gare, période ou type de train


## Stack technique

Outil: 
    -`pandas` / `numpy`: Manipulation des données
    -`matplotlib` / `seaborn`: Visualisations
    -`scikit-learn`: Modélisation et évaluation
    -`streamlit`: Dashboard interactif
    -`ruff`: Formatage du code 

