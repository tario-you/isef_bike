import tensorflow
import rl
import keras
import random
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Embedding
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_model(actions):
    model = Sequential()
    # model.add(Flatten(input_shape=(1,3))) # vel, wat, rpm
    model.add(Embedding(input_dim=150, output_dim=10))
    model.add(LSTM(24))
    model.add(Flatten())
    model.add(Dense(10))
    # model.add(Dense(24, activation="relu"))
    # model.add(Dense(24, activation="relu"))
    # model.add(Dense(24, activation="relu"))
    # model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


if __name__ == "__main__":
    actions = [-1, 0, 1]
    model = build_model(len(actions))
    # print(model.summary())
    dqn = build_agent(model, len(actions))
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    dqn.fit(nb_steps=50000, visualize=False, verbose=1)
