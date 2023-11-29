import pandas as pd
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

class TradingEnv(Env):
    def __init__(self, df):
        self.df = df
        self.balance = 10000
        self.net_worth = []
        
        # Dimensions des espaces d'observations et d'actions
        self.observation_space = Box(low=0, high=np.inf, shape=(6,)) 
        self.action_space = Discrete(3)
        
        # Episode
        self._start_tick = 0
        self._end_tick = len(df) - 1
        self._done = False
        self._current_tick = 0
        
        self.reset()
        
    def reset(self):
        # Réinitialiser l'environnement
        self._current_tick = self._start_tick
        self._done = False
        self.balance = 10000
        self.net_worth = [self.balance]
        self.position = 0
        
        return self._get_observation() 
    
    def step(self, action):
       # Actions: 0=Acheter, 1=Vendre, 2=Rien faire
       
       current_price = self._get_current_close()
        
       if action == 0:
           # Acheter 
           qty = 1000 // current_price
           self.balance -= qty * current_price
           self.position += qty
       elif action == 1:
           # Vendre
           qty = min(abs(self.position), 1000 // current_price)  
           self.balance += qty * current_price
           self.position -= qty
           
       # Calculer reward 
       self.net_worth.append(self.balance + self.position * current_price) 
       reward = self.net_worth[-1] - self.net_worth[-2]
       
       # MAJ état 
       self._current_tick += 1
       self._done = self._current_tick == self._end_tick
        
       return self._get_observation(), reward, self._done, {}
    
    def _get_observation(self):
        obs = np.array([
            self.balance,
            self.position,
            self._get_current_open(), 
            self._get_current_high(),
            self._get_current_low(), 
            self._get_current_close()
        ])
        
        return obs
    
    def _get_current_close(self):
        return self.df.loc[self._current_tick, 'Close']
    
    def _get_current_open(self):
        return self.df.loc[self._current_tick, 'Open']

    def _get_current_high(self):
        return self.df.loc[self._current_tick, 'High']

    def _get_current_low(self):
        return self.df.loc[self._current_tick, 'Low']