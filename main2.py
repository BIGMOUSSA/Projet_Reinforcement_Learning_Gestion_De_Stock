# main.py

import gym
import numpy as np
#from stable_baselines3 import PPO
from DQNagent import Agent
from utils import formatPrice, getStockDataVec, sigmoid, getState
#from trainding_env import TradingEnv

# User input
stock_name = input("Enter stock_name")
window_size = 2 #int(input())
episode_count = 10 #int(input())
stock_name = "cleaned_data/"+str(stock_name)

# Reinforcement Learning Agent
agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []

    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # Buy
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0:  # Sell
            bought_price = window_size_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 10 == 0:
        agent.model.save(str(e))

# Reinforcement Learning with Stable Baselines
#class StockTradingEnv(gym.Env):
    # Implementation of the environment (fill in as needed)

#env = StockTradingEnv()
#model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=10000)
#model.save("ppo_stock")
