
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Define Supply Chain Environment
class SupplyChainEnv(gym.Env):
    def __init__(self):
        super(SupplyChainEnv, self).__init__()
        self.max_inventory = 100
        self.max_demand = 50
        self.storage_cost = 2
        self.stockout_penalty = 10
        self.order_cost = 5

        self.state = [50, 0]  # [inventory_level, demand]
        self.action_space = gym.spaces.Discrete(11)  # Order 0 to 10 units
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), high=np.array([self.max_inventory, self.max_demand]), dtype=np.int32)

    def step(self, action):
        order_quantity = action * 10
        inventory_level, demand = self.state
        received = order_quantity
        new_inventory = inventory_level + received - demand

        if new_inventory < 0:
            penalty = abs(new_inventory) * self.stockout_penalty
            new_inventory = 0
        else:
            penalty = 0

        holding_cost = new_inventory * self.storage_cost
        order_expense = self.order_cost * (order_quantity > 0)
        reward = - (holding_cost + order_expense + penalty)

        self.state = [new_inventory, np.random.randint(10, self.max_demand)]
        done = False
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = [50, np.random.randint(10, self.max_demand)]
        return np.array(self.state)

    def render(self):
        print(f"Inventory: {self.state[0]}, Demand: {self.state[1]}")

# Train Supply Chain Agent with PPO
def train_supply_chain_ai():
    env = SupplyChainEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("supply_chain_agent")
    return model

# Simulate Supply Chain Decision Making
def simulate_supply_chain(model, episodes=20):
    env = SupplyChainEnv()
    state = env.reset()
    rewards = []

    for _ in range(episodes):
        action, _ = model.predict(state)
        state, reward, _, _ = env.step(action)
        rewards.append(reward)

    plt.plot(rewards)
    plt.xlabel("Time Steps")
    plt.ylabel("Reward")
    plt.title("Supply Chain Optimization Performance")
    plt.savefig("supply_chain_rewards.png")
    plt.close()

if __name__ == "__main__":
    print("Training AI for supply chain optimization...")
    trained_model = train_supply_chain_ai()
    print("Training complete. Running simulations...")
    simulate_supply_chain(trained_model)
    print("Simulation complete. AI model is ready for deployment.")
