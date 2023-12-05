import gym
import numpy as np

from DQNagent import Agent
from utils import formatPrice, getStockDataVec, sigmoid, getState
from tqdm import tqdm

window_size = 1  
episode_count = int(input(" Enter the number of episode : "))
test = True
while test :
    choice = input("Quel fichier voulez-vous entrainer : 1 = Training.csv, 2 = test.csv) :")
    try :
        choice = int(choice)
    except :
        print("Choice doit être 1 ou 2")
    if choice == 1 :
        stock_name = "cleaned_data/Training"
        test = False
    elif choice ==2 :
        stock_name = "cleaned_data/test"
        test = False
    else :
        print("Choix invalide")

# Reinforcement Learning Agent
agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 64

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []

    for t in tqdm(range(l), total = len(range(l))):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        print(f" \n étape  {t} sur {l}")
        if action == 1:  # Buy
            agent.inventory.append(data[t])
            print("Achète: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0:  # Sell
            bought_price = window_size_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Vend : " + formatPrice(data[t]) + " | Bénéfice: " + formatPrice(data[t] - bought_price))

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("---------------  Stock Trading -----------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("----------------- Stock Trading ---------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 10 == 0:
        agent.model.save(str(e))
