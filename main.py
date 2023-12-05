
import pandas as pd
from env import TradingEnv
from agent import TradingQAgent


# Charger les données
data = pd.read_csv('cleaned_data/training.csv', index_col=0)



tradingenv = TradingEnv(data)

Tradingagent = TradingQAgent(tradingenv)

Tradingagent.train(5)
