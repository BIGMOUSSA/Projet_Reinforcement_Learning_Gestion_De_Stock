import pandas as pd
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

class TradingEnv(Env):
    def __init__(self, df):
        self.df = df
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.net_worth = []
        self.max_position = 1000  # Maximum position size
        
        # Dimensions des espaces d'observations et d'actions
        self.observation_space = Box(low=0, high=1, shape=(6,))  # Normalize to [0, 1] range
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
        self.balance = self.initial_balance
        self.net_worth = [self.balance]
        self.position = 0

        return self._get_observation()

    def step(self, action):
        # Actions: 0=Acheter, 1=Vendre, 2=Rien faire

        current_price = self._get_current_close()
        qty = 0
        if action == 0:
            # Acheter
            qty = min(self.max_position // current_price, self.balance // current_price)
            cost = qty * current_price
            self.balance -= cost
            self.position += qty
        elif action == 1:
            # Vendre
            qty = min(abs(self.position), self.max_position)
            revenue = qty * current_price
            self.balance += revenue
            self.position -= qty

        # Consider transaction costs (you can adjust the transaction cost as needed)
        transaction_cost = 0.001 * abs(qty * current_price)
        self.balance -= transaction_cost

        # Calculate reward
        self.net_worth.append(self.balance + self.position * current_price)
        reward = self.net_worth[-1] - self.net_worth[-2]

        # Update state
        self._current_tick += 1
        self._done = self._current_tick == self._end_tick

        return self._get_observation(), reward, self._done, {}

    def _get_observation(self):
        obs = np.array([
            self.balance / self.initial_balance,  # Normalize to [0, 1]
            self.position / self.max_position,  # Normalize to [0, 1]
            self._get_current_open() / 100,  # Normalize to [0, 1]
            self._get_current_high() / 100,  # Normalize to [0, 1]
            self._get_current_low() / 100,  # Normalize to [0, 1]
            self._get_current_close() / 100  # Normalize to [0, 1]
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
