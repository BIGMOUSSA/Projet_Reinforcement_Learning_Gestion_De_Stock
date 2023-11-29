import numpy as np
import pandas as pd

class TradingQAgent:
    def __init__(self, env):
        self.env = env
        self.lr = 0.1 
        self.gamma = 0.95
        self.eps = 1.0
        self.decay = 0.99995
        self.q_table = pd.DataFrame(columns=list(range(env.action_space.n)), 
                                    dtype=np.float64)
                                    
    def train(self, episodes):
        for e in range(episodes):
            state = self.env.reset()
            done = False
            score = 0
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.update_qtable(state, action, reward, next_state)
                
                state = next_state
                score += reward
                
            self.eps = max(0.01, self.eps*self.decay)        
            print("Episode {} Score {}".format(e,score))
            
            
    def get_action(self, state):
        if np.random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table.loc[tuple(state)])     
            
            
    def update_qtable(self, state, action, reward, next_state):
        q_1 = self.q_table.loc[tuple(state)][action]
        q_2 = reward + self.gamma*max(self.q_table.loc[tuple(next_state)])
        self.q_table.loc[tuple(state)][action] += self.lr*(q_2 - q_1)  
        
    def test(self, episodes):
        for e in range(episodes):
            done = False
            score = 0
            state = self.env.reset()
            
            while not done:
                action = np.argmax(self.q_table.loc[tuple(state)])
                state, reward, done, _ = self.env.step(action)
                
                score += reward
                
            print("Test {} Score {}".format(e, score))
  