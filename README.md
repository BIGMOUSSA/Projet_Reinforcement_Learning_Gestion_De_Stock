# Stock Prediction System
Mise en œuvre de l'apprentissage par renforcement Q-learning pour le trading à court terme sur actions. Le modèle détermine si la meilleure action à prendre est de rester en position, de vendre ou d'acheter.

# Dataset
Le jeu de données utilisé est disponible ici : https://github.com/BIGMOUSSA/Projet_Reinforcement_Learning_Gestion_De_Stock/tree/main/cleaned_data

# Plan de travail

## Preporocessing du dataset

1. **Chargement des Données :**
   Les données sont chargées à partir de fichiers CSV contenant les informations du marché boursier, notamment les colonnes de date, d'ouverture, de fermeture, de plus haut, de plus bas et de volume.

2. **Conversion des Colonnes de Date :**
   La colonne de date est convertie en format datetime pour faciliter la manipulation temporelle.

3. **Vérification des Valeurs Manquantes :**
   Une vérification est effectuée pour détecter la présence de valeurs manquantes dans les données. Aucune donnée manquante n'a été identifiée dans l'échantillon d'entraînement ou de test.

4. **Suppression de Colonnes Redondantes :**
   Une analyse comparative révèle que la colonne "Close" et la colonne "Adj Close" sont identiques. Par conséquent, la colonne "Adj Close" est supprimée pour simplifier les données.

5. **Visualisation des Données :**
   Des graphiques de dispersion sont créés pour visualiser l'évolution du volume, des prix d'ouverture, de fermeture, du plus haut et du plus bas au fil du temps.

6. **Enregistrement des Données Nettoyées :**
   Les données nettoyées sont enregistrées dans de nouveaux fichiers CSV ("cleaned_data/Training.csv" et "cleaned_data/test.csv") pour une utilisation ultérieure dans l'analyse financière.


## Définir l’environnement
En apprentissage par renforcement (Reinforcement Learning), un environnement est généralement défini comme le cadre dans lequel un agent interagit pour apprendre. L'environnement représente le monde dans lequel l'agent opère, et il est caractérisé par les états possibles dans lesquels l'agent peut se trouver, les actions que l'agent peut entreprendre, les récompenses associées à certaines transitions état-action, et éventuellement les probabilités de transition entre les états.

Pour le trading à court terme sur actions, l'environnement d'apprentissage par renforcement peut être défini de la manière suivante :

**Etats (S)** :
- Comptes de trading (solde monétaire actuel, profits/pertes cumulés)
- Données de marché (prix des devises, volumes de trading, indicateurs techniques, etc.)
- Positions ouvertes
- Informations économiques et géopolitiques 

**Actions (A)** :
- Acheter / Vendre une certaine quantité d'une devise
- Fermer ou ajuster une position ouverte  
- Ne rien faire

**Récompenses (R)** :  
- Variation de la valeur nette du compte de trading (profits/pertes réalisés)
- Commissions, spreads, frais de transactions

**Transitions (P)** : 
- Evolution stochastique des prix des devises en fonction des actions de l'agent et des autres acteurs du marché

**Observations (O)** :
- Données de marché visibles par l'agent (taille de l'historique, granularité, etc.)

Ainsi l'objectif de l'agent sera de maximiser ses profits en trading sur le marché forex de manière automatisée en apprenant à effectuer les meilleures actions en fonction des états observés.

## Définir l'agent
Dans le contexte de l'apprentissage par renforcement (Reinforcement Learning), un agent désigne l'entité qui va interagir avec l'environnement pour apprendre à résoudre une tâche.

Les principales caractéristiques d'un agent de RL sont :

- Il peut percevoir l'état courant de l'environnement (observations)
- Il peut choisir et exécuter des actions qui vont influer sur l'environnement 
- Il a pour objectif d'apprendre grâce aux récompenses reçues à maximiser ses performances à long terme (cumul de récompenses)

Dans notre cas:

- L'environnement est le marché financier simulé par la classe `ForexEnv` 
- L'agent est `ForexDQNAgent` ou `ForexQAgent`
- A chaque pas de temps, l'agent:
    - Observe l'état courant (cours des devises, solde etc.)
    - Choisi une action (acheter, vendre, ne rien faire)
    - Reçoit une récompense (variation de la valeur nette)
    
Au fur et à mesure, l'agent apprend la politique optimale, c'est à dire quelle action choisir dans chaque état observé pour maximiser ses profits.

En résumé, l'agent est l'entité autonome qui apprend par essais-erreurs à interagir de façon optimale dans l'environnement.

## Entrainement du modèle (Q-learning)

Ce code a été développé pour entraîner un agent de trading utilisant l'apprentissage par renforcement (Q-learning) sur des données du marché financier. L'agent est conçu pour prendre des décisions d'achat, de vente ou de maintien de position en fonction de son apprentissage sur les données historiques.

1. **Chargement des Données :**
    Les données d'entraînement sont chargées à partir du fichier CSV "cleaned_data/training.csv". 

2. **Initialisation de l'Environnement :**
   Créez une instance de l'environnement de trading en utilisant la classe `TradingEnv` fournie. L'environnement prend les données chargées comme argument.

   ```python
   import pandas as pd
   from env import TradingEnv
   from agent import TradingQAgent

   # Charger les données
   data = pd.read_csv('cleaned_data/training.csv', index_col=0)

   # Initialiser l'environnement
   tradingenv = TradingEnv(data)
   ```

3. **Initialisation de l'Agent :**
   Créez une instance de l'agent de trading en utilisant la classe `TradingQAgent`. Vous pouvez ajuster les hyperparamètres de l'agent selon vos besoins.

   ```python
   # Initialiser l'agent
   Tradingagent = TradingQAgent(tradingenv)
   ```

4. **Entraînement de l'Agent :**
   Lancez l'entraînement de l'agent en utilisant la méthode `train`. Spécifiez le nombre d'épisodes d'entraînement que vous souhaitez exécuter.

   ```python
   # Entraîner l'agent
   Tradingagent.train(100)
   ```

5. **Test de l'Agent :**
   Vous pouvez évaluer les performances de l'agent en utilisant la méthode `test`. Spécifiez le nombre d'épisodes de test que vous souhaitez exécuter.

   ```python
   # Tester l'agent
   Tradingagent.test(5)
   ```