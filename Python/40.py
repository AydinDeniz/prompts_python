
import ccxt
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

# Initialize exchange (Binance)
exchange = ccxt.binance({
    'rateLimit': 1200,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

symbol = 'BTC/USDT'
balance = 10000  # Initial balance in USD
position = 0  # Number of BTC held

# Define a trading environment
class CryptoTradingEnv(gym.Env):
    def __init__(self):
        super(CryptoTradingEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.current_step = 0
        self.balance = balance
        self.position = position
        self.data = self.fetch_market_data()

    def fetch_market_data(self):
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=100)
        return np.array([entry[4] for entry in ohlcv])  # Closing prices

    def step(self, action):
        price = self.data[self.current_step]
        reward = 0

        if action == 0 and self.balance >= price:  # Buy
            self.position += 1
            self.balance -= price
        elif action == 2 and self.position > 0:  # Sell
            self.balance += price
            self.position -= 1
            reward = self.balance - balance

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        obs = np.array([self.balance, self.position, price, np.mean(self.data), np.std(self.data), self.current_step])
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        self.balance = balance
        self.position = position
        return np.array([self.balance, self.position, self.data[0], np.mean(self.data), np.std(self.data), self.current_step])

# Train trading bot
env = DummyVecEnv([lambda: CryptoTradingEnv()])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("crypto_trading_bot")

print("Model trained and saved as 'crypto_trading_bot.zip'.")
