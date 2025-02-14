
import numpy as np
import pybullet as p
import pybullet_data
import time
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from stable_baselines3 import PPO

# Initialize Underwater Simulation
class UnderwaterEnv(gym.Env):
    def __init__(self):
        super(UnderwaterEnv, self).__init__()
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.submarine = p.loadURDF("submarine.urdf", [0, 0, -5], useFixedBase=False)
        self.target = np.array([random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 0)])

        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)

    def step(self, action):
        if action == 0:
            p.applyExternalForce(self.submarine, -1, [1, 0, 0], [0, 0, 0], p.WORLD_FRAME)
        elif action == 1:
            p.applyExternalForce(self.submarine, -1, [-1, 0, 0], [0, 0, 0], p.WORLD_FRAME)
        elif action == 2:
            p.applyExternalForce(self.submarine, -1, [0, 1, 0], [0, 0, 0], p.WORLD_FRAME)
        elif action == 3:
            p.applyExternalForce(self.submarine, -1, [0, -1, 0], [0, 0, 0], p.WORLD_FRAME)
        elif action == 4:
            p.applyExternalForce(self.submarine, -1, [0, 0, 1], [0, 0, 0], p.WORLD_FRAME)
        elif action == 5:
            p.applyExternalForce(self.submarine, -1, [0, 0, -1], [0, 0, 0], p.WORLD_FRAME)

        p.stepSimulation()
        time.sleep(0.02)

        sub_pos, _ = p.getBasePositionAndOrientation(self.submarine)
        distance = np.linalg.norm(np.array(sub_pos) - self.target)

        reward = -distance
        done = distance < 0.5
        obs = np.concatenate((np.array(sub_pos), self.target))

        return obs, reward, done, {}

    def reset(self):
        p.resetSimulation()
        self.submarine = p.loadURDF("submarine.urdf", [0, 0, -5], useFixedBase=False)
        self.target = np.array([random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 0)])
        return np.concatenate((np.array([0, 0, -5]), self.target))

    def render(self):
        pass

    def close(self):
        p.disconnect()

# Train AI Model for Navigation
def train_underwater_ai():
    env = UnderwaterEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("underwater_navigation_model")

# Simulate Sonar Data
def simulate_sonar(submarine_position, obstacles):
    sonar_readings = []
    for obs in obstacles:
        distance = np.linalg.norm(np.array(submarine_position) - np.array(obs))
        sonar_readings.append(distance)
    return np.array(sonar_readings)

if __name__ == "__main__":
    print("Training AI for underwater navigation...")
    train_underwater_ai()
    print("Model training complete. Ready for deployment.")
