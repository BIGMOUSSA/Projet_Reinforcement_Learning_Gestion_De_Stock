
import pandas as pd
from env import TradingEnv
from agent import TradingQAgent
import argparse


parser = argparse.ArgumentParser(description = "Application pour le trading stock par reinforcement pour un dataset donné")
parser.add_argument("--episode", help = "nombre d'épisode", default=10, type = int)
parser.add_argument("--path_data", default="cleaned_data/training.csv")
args = parser.parse_args()
# Charger les données
try :
    data = pd.read_csv(args.path_data, index_col=0)
except Exception as e :
    print(e , "donne le chemin d'accès valide ou laisse vide")



tradingenv = TradingEnv(data)

Tradingagent = TradingQAgent(tradingenv)

try :
    Tradingagent.train(args.episode)
except Exception as e :
    print(e , "le nombre d'épisode doit être un entier")
