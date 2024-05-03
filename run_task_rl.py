import gym
import task_rl
import numpy as np
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.optimizers.legacy import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from rl.core import Processor

class CustomProcessor(Processor):
    def process_observation(self, observation):
        # Extract agent and target positions and flatten them into a single array
        agent_coords = observation["agent"]
        target_coords = observation["target"]
        # Concatenate and reshape to ensure it fits the expected input shape for the network
        return np.concatenate([agent_coords, target_coords]).reshape(1, 4)

    def process_state_batch(self, batch):
        # Batch processing if necessary
        return np.array(batch).reshape(-1, 1, 4)


def run_agent_tests(env, agent, episodes):
    results = agent.test(env, nb_episodes=episodes, visualize=True, verbose=0)
    print(results.history)
    

def main_menu():
    print("Welcome to the Drone Navigation Environment Menu!")
    print("1. Execute a random navigation strategy with the Drone")
    print("2. Execute a pre-trained DQN (Deep Q-Network) navigation strategy with the Drone")
    print("3. Train a DQN navigation agent within the Drone Navigation Environment")
    print("4. Exit")

def run_random_strategy(env, episodes):
    print("Executing a random navigation strategy with the Drone...")
    for episode in range(1, episodes + 1):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            observation, reward, done, info = env.step(env.action_space.sample())
            score += reward 

        print(f"Episode {episode} Score {score}")  


def run_pretrained_strategy(env, agent, episodes):
    print("Executing a pre-trained DQN navigation strategy with the Drone...")
    agent.load_weights('dqn_wights.h5f')
    run_agent_tests(env, agent, episodes)


def train_navigation_strategy(env, agent, episodes):
    print("Training a DQN navigation agent within the Drone Navigation Environment...")
    agent.fit(env, nb_steps=50000, visualize=False, verbose=1)
    agent.save_weights('drone_navigation_weights.h5f', overwrite=True)
   
    run_agent_tests(env, agent, episodes)

def exit_program(env):
    print("Exiting the Drone Navigation Environment...")
    env.close()
    exit()

def main():
    grid_size=4
    max_steps=1000
    start_position=(1,1)
    env = gym.make('task_rl/DroneNavigation-v0', render_mode="human", size=grid_size, start_position=start_position, max_steps=max_steps)
    env.action_space.seed(42)
    actions = env.action_space.n    
    episodes = 10
    
    model = Sequential()
    model.add(Flatten(input_shape=(1, 4)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))

    agent = DQNAgent(
        model=model,
        processor=CustomProcessor(),
        memory=SequentialMemory(limit=50000, window_length=1),
        policy=BoltzmannQPolicy(),
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=0.01
    )

    agent.compile(Adam(lr=1e-3), metrics=["mae"])
    

    while True:
        main_menu()
        choice = input("Please enter your selection: ")

        if choice == '1':
            run_random_strategy(env, episodes)
        elif choice == '2':
            run_pretrained_strategy(env, agent, episodes)
        elif choice == '3':
            train_navigation_strategy(env, agent, episodes)
        elif choice == '4':
            exit_program(env)
        else:
            print("Invalid selection. Please enter a number from 1 to 4.")
    

if __name__ == "__main__":
    main()

