# Stock Prediction System
Mise en œuvre de l'apprentissage par renforcement Q-learning pour le trading à court terme sur actions. Le modèle détermine si la meilleure action à prendre est de rester en position, de vendre ou d'acheter.

# Dataset
Le jeu de données utilisé est disponible ici : https://github.com/BIGMOUSSA/Projet_Reinforcement_Learning_Gestion_De_Stock/tree/main/cleaned_data

# Plan de travail
## Preporocessing du dataset
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





Voici quelques éléments clés pour définir un environnement en Reinforcement Learning :

**États (States)** : Les états représentent les informations nécessaires pour décrire la situation actuelle du marché Forex. Cela peut inclure des données historiques sur les prix (cours d'ouverture, de clôture, hauts, bas), les volumes de transactions, les indicateurs techniques (moyennes mobiles, RSI, MACD, etc.), et d'autres caractéristiques pertinentes.

**Actions** : Les actions sont les décisions que l'agent peut prendre sur le marché Forex. Cela peut inclure des actions telles que "acheter", "vendre", "rester hors du marché", ou des actions plus complexes basées sur des combinaisons d'instruments financiers.

**Récompenses (Rewards)** : Les récompenses représentent la performance de l'agent après avoir entrepris une action particulière. Dans le contexte du trading Forex, la récompense peut être définie en fonction du profit ou de la perte résultant de l'action de l'agent.

**Politique** : La politique est la stratégie que l'agent suit pour prendre des décisions. Dans le trading Forex, cela peut être une fonction qui mappe les états aux actions, basée sur des algorithmes d'apprentissage automatique.

**Fonction de transition** : La fonction de transition définit comment l'état du marché évolue en réponse aux actions de l'agent. Cela pourrait être basé sur les changements réels dans les prix des paires de devises.

**Horizon temporel** : Il s'agit de la durée d'une transaction ou d'une série de transactions que l'agent considère avant de réévaluer sa stratégie.

**Terminaison** : Les conditions sous lesquelles une transaction ou une série de transactions est considérée comme terminée. Cela pourrait être basé sur une limite de temps, un objectif de profit atteint, ou une perte maximale tolérée.

## Définir l'agent
## Entrainement du modèle ( Q-learning , tensorflow)
## Evaluation du modèle
## Optimisation  des paramètres