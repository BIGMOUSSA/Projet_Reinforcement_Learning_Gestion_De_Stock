# Stock Prediction System - README

## Prétraitement du jeu de données

### Chargement des Données

Les données du marché boursier sont chargées à partir de fichiers CSV disponibles [ici](https://github.com/BIGMOUSSA/Projet_Reinforcement_Learning_Gestion_De_Stock/tree/main/cleaned_data). Ces fichiers comprennent des informations telles que la date, l'ouverture, la fermeture, le plus haut, le plus bas et le volume.

### Conversion des Colonnes de Date

La colonne de date est convertie en format datetime pour simplifier la manipulation temporelle.

### Vérification des Valeurs Manquantes

Une vérification est effectuée pour détecter les valeurs manquantes. Aucune donnée manquante n'a été identifiée dans l'échantillon d'entraînement ou de test.

### Suppression de Colonnes Redondantes

Une analyse comparative révèle que les colonnes "Close" et "Adj Close" sont identiques. La colonne "Adj Close" est supprimée pour simplifier les données.

### Visualisation des Données

Des graphiques de dispersion sont créés pour visualiser l'évolution du volume, des prix d'ouverture, de fermeture, du plus haut et du plus bas au fil du temps.

### Enregistrement des Données Nettoyées

Les données nettoyées sont enregistrées dans de nouveaux fichiers CSV ("cleaned_data/Training.csv" et "cleaned_data/test.csv") pour une utilisation ultérieure dans l'analyse financière.

## Installation des Dépendances

Installez les dépendances nécessaires en exécutant la commande suivante :

```bash
pip install -r requirements.txt
```

Cela garantira que toutes les bibliothèques requises sont correctement installées avant d'exécuter les scripts.

## Définition de l'environnement

En apprentissage par renforcement, l'environnement est défini comme le cadre dans lequel l'agent interagit pour apprendre. Pour le trading à court terme sur actions, l'environnement est caractérisé par les états (comptes de trading, données de marché, positions ouvertes, informations économiques), les actions possibles (acheter, vendre, ne rien faire), les récompenses (variation de la valeur nette), les transitions (évolution stochastique des prix) et les observations (données de marché visibles par l'agent).

L'objectif de l'agent est de maximiser ses profits en apprenant à effectuer les meilleures actions en fonction des états observés.

## Définition de l'Agent

En apprentissage par renforcement, l'agent est l'entité qui interagit avec l'environnement pour apprendre à résoudre une tâche. Les caractéristiques principales d'un agent de RL sont sa capacité à percevoir l'état actuel de l'environnement, choisir et exécuter des actions, et apprendre grâce aux récompenses reçues pour maximiser ses performances à long terme.

Deux solutions sont proposées :

1. **Q-Learning avec Exploration-Exploitation (agent.py)** : Utilise le Q-Learning en explorant l'environnement, mappant 'state' et 'q_value', et combinant exploration et exploitation. Adapté aux environnements petits et discrets. Pour tester, exécutez **main.py** en spécifiant le nombre d'épisodes et/ou le chemin du fichier. Options par défaut : `python main.py` ou `python main.py --episode 10 --path_data "cleaned_data/Training.csv"`.
```bash
python main.py
```
ou
```bash
python main.py --episode 10 --path_data "cleaned_data/Training.csv"
```

2. **DQN Agent (DQNagent.py)** : Utilise les réseaux de neurones pour l'entraînement. Lancez **main2.py**, choisissez le nombre d'épisodes et optez pour le fichier d'entraînement (option 1) ou de test (option 2). Exemple : `python main2.py`.
```bash
python main2.py
```
L'application vous demandera le nombre d'épisodes pour l'entraînement et le choix du fichier (option 1 pour l'entraînement, option 2 pour le test).