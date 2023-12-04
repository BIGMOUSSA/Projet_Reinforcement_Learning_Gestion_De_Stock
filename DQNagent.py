import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


class ForexDQNAgent:
    def __init__(self, env):
        self.env = env

        # Paramètres DQN
        self.learning_rate = 0.001
        
        # Création du modèle DQN
        self.model = self.create_model()  
        
        # Politique d'exploration de Boltzmann
        self.policy = BoltzmannQPolicy()
        
        # Mémoire séquentielle
        self.memory = SequentialMemory(limit=1000, window_length=1)
        
        # Configuration de l'agent DQN
        self.dqn_agent = DQNAgent(model=self.model, 
                                   memory=self.memory,
                                   policy=self.policy, 
                                   nb_actions=self.env.action_space.n,
                                   nb_steps_warmup=100,
                                   target_model_update=1e-2,
                                   enable_double_dqn=True)
        self.dqn_agent.compile(Adam(learning_rate=self.learning_rate))
        
    def create_model(self):
       # Création du réseau de neurones 
       model = Sequential()
       model.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
       model.add(Dense(16, activation='relu'))
       model.add(Dense(16, activation='relu'))
       model.add(Dense(self.env.action_space.n, activation='linear'))
       
       return model
       
    def train(self, episodes):
       for e in range(episodes):
           state = self.env.reset()
           done = False
           score = 0 
           
           while not done:
               action = self.dqn_agent.forward([state])  
               next_state, reward, done, _ = self.env.step(action)
               
               self.dqn_agent.step(state, action, reward, next_state, done)  
               
               state = next_state
               score += reward
               print("Score ", score)
               
           print("Episode :", e+1, "Score :", score)
           
    def test(self, episodes):
       for e in range(episodes):
            done = False
            score = 0
            state = self.env.reset()
            
            while not done:
                action = np.argmax(self.dqn_agent.model.predict(np.array([state])))
                state, reward, done, _ = self.env.step(action)
                
                score += reward
                
            print("Test {} Score {}".format(e, score))
        
