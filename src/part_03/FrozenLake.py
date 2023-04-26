from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from QLearning import QLearning
from numpy import loadtxt
from SARSA import Sarsa
import warnings
warnings.simplefilter("ignore")

# exemplo de ambiente nao determinístico
env = gym.make('FrozenLake-v1', map_name="8x8", render_mode='ansi').env

# only execute the following lines if you want to create a new q-table
# qlearn = QLearning(env, alpha=0.1, gamma=0.99, epsilon=1, epsilon_min=0.0091, epsilon_dec=0.9999, episodes=70000)
# q_table = qlearn.train('data/q-table-frozen-lake-qlearning.csv','results/frozen_lake_qlearning')
# q_table = loadtxt('data/q-table-frozen-lake-qlearning.csv', delimiter=',')

sarsa = Sarsa(env, alpha=0.1, gamma=0.99, epsilon=1, epsilon_min=0.0091, epsilon_dec=0.9999, episodes=70000)
q_table = sarsa.train('data/q-table-frozen-lake-sarsa.csv','results/frozen_lake_sarsa')
# env = gym.make('FrozenLake-v1', render_mode='human').env

# (state, _) = env.reset()
# epochs = 0
# rewards = 0
# done = False
    
# while not done:
#     action = np.argmax(q_table[state])
#     state, reward, done, _, info = env.step(action)
#     epochs += 1
#     rewards += reward

# print("\n")
# print("Timesteps taken: {}".format(epochs))
# print("Rewards: {}".format(rewards))