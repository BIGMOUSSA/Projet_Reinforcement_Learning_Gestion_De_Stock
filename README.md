# Stock Prediction System
Mise en œuvre de l'apprentissage par renforcement Q-learning pour le trading à court terme sur actions. Le modèle détermine si la meilleure action à prendre est de rester en position, de vendre ou d'acheter.

# Dataset
Le jeu de données utilisé est disponible ici : https://github.com/BIGMOUSSA/Projet_Reinforcement_Learning_Gestion_De_Stock/tree/main/cleaned_data

# Plan de travail

## Preporocessing du dataset
Le prétraitement a consisté à vérifier si le fichier contenait des doublons et à examiner les relations entre certaines variables. Nous avons constaté que les deux fichiers mis à notre disposition ne contenaient pas de doublons. Nous avons également remarqué que la variable "Adj_close" contenait exactement les mêmes valeurs que "Close", nous l'avons donc supprimée. À la fin du prétraitement, nous avons créé un dossier appelé "Cleaned_data" contenant deux fichiers, 'Training.csv' et 'Testing.csv', qui seront ensuite utilisés pour l'entraînement et l'évaluation du modèle.
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

## Entrainement du modèle ( Q-learning , tensorflow)
Pour entrainer le model, il su
## Evaluation du modèle

## Optimisation  des paramètres