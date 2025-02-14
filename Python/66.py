
import airsim
import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from collections import deque

# Connect to AirSim
class UAVSwarmEnv:
    def __init__(self, num_drones=3):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.num_drones = num_drones
        self.drones = [f"Drone{i}" for i in range(num_drones)]
        self.target = np.array([random.uniform(-50, 50), random.uniform(-50, 50), random.uniform(-10, 10)])
        self.reset()

    def reset(self):
        print("Resetting UAV Swarm Environment...")
        for drone in self.drones:
            self.client.reset()
            self.client.enableApiControl(True, drone)
            self.client.armDisarm(True, drone)
            self.client.takeoffAsync(vehicle_name=drone).join()
        return self.get_state()

    def get_state(self):
        state = []
        for drone in self.drones:
            pose = self.client.simGetVehiclePose(vehicle_name=drone)
            state.append([pose.position.x_val, pose.position.y_val, pose.position.z_val])
        return np.array(state).flatten()

    def step(self, actions):
        for i, drone in enumerate(self.drones):
            move_x, move_y, move_z = actions[i]
            self.client.moveByVelocityAsync(move_x, move_y, move_z, 1, vehicle_name=drone)

        time.sleep(1)
        new_state = self.get_state()
        reward = -np.linalg.norm(new_state - self.target)
        done = np.linalg.norm(new_state - self.target) < 5
        return new_state, reward, done, {}

# Train AI Controller for UAV Swarm
def train_uav_swarm():
    env = UAVSwarmEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("uav_swarm_model")

# Deploy UAV Swarm in Simulation
def deploy_uav_swarm():
    env = UAVSwarmEnv()
    model = PPO.load("uav_swarm_model")

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

if __name__ == "__main__":
    print("Training AI for UAV swarm control...")
    train_uav_swarm()
    print("Training complete. Deploying UAV swarm...")
    deploy_uav_swarm()
    print("UAV swarm deployment complete.")
