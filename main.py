
import pandas as pd
from env import TradingEnv
from agent import TradingQAgent

# Charger les données
data = pd.read_csv('url to data')
data.drop(columns={'Unnamed: 0'}, inplace=True)
# data.head()


tradingenv = TradingEnv(data)
Tradingagent = TradingQAgent(tradingenv)

Tradingagent.train(100)
Tradingagent.test(10)