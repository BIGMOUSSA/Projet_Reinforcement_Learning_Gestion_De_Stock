import tensorflow as tf

class TradingAgent(object):

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        self.q_network = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear')
        ])

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) > self.batch_size:
            self.train()

        if not done:
            next_action = self.act(next_state)
            self.target_q = self.q_network(next_state).numpy()[next_action]
        else:
            self.target_q = reward

        return self.q_network(state).numpy()[action]

    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_network(state).numpy())

    def train(self):
        batch = np.array(self.memory)
        state_batch = batch[:, 0]
        action_batch = batch[:, 1]
        reward_batch = batch[:, 2]
        next_state_batch = batch[:, 3]
        done_batch = batch[:, 4]

        target_q = self.target_q
        with tf.GradientTape() as tape:
            q_values = self.q_network(state_batch)
            loss = tf.keras.losses.mse(target_q, q_values)

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.q_network.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        self.memory = []
