
import pandas as pd
from env import TradingEnv
from agent import TradingQAgent
#from DQNagent import ForexDQNAgent

# Charger les données
data = pd.read_csv('cleaned_data/training.csv', index_col=0)
#data.drop(columns={'Unnamed: 0'}, inplace=True)
#print(data.head())


tradingenv = TradingEnv(data)
#print(TradingQAgent(tradingenv).q_table)
Tradingagent = TradingQAgent(tradingenv)
#DQN_traiding = ForexDQNAgent(tradingenv)
#DQN_traiding.train(10)
Tradingagent.train(5)
#Tradingagent.test(10)