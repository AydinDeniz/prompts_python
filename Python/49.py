
import openai
import random
import json
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from collections import defaultdict

# OpenAI API key
API_KEY = "your_openai_api_key"

# Load pre-trained chatbot model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define reinforcement learning parameters
Q_TABLE = defaultdict(lambda: np.zeros(3))  # Actions: (respond, escalate, clarify)
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1  # Exploration rate

# Reward system
REWARD_MAPPING = {"positive": 10, "neutral": 0, "negative": -10}

# Simulated user feedback (for training)
def get_user_feedback(response):
    feedback_options = ["positive", "neutral", "negative"]
    return random.choice(feedback_options)

# AI chatbot response using GPT-4
def generate_ai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful customer support assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response["choices"][0]["message"]["content"]

# Train chatbot using reinforcement learning
def train_chatbot(training_data, episodes=100):
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes} - Training in progress...")
        for query in training_data:
            state = query
            action = np.argmax(Q_TABLE[state]) if random.uniform(0, 1) > EPSILON else random.choice(range(3))
            response = generate_ai_response(state)
            feedback = get_user_feedback(response)
            reward = REWARD_MAPPING[feedback]

            next_state = query + " " + response  # Simulating next state
            best_future_q = np.max(Q_TABLE[next_state])

            # Update Q-table
            Q_TABLE[state][action] = (1 - LEARNING_RATE) * Q_TABLE[state][action] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_future_q)

    print("Training complete. Q-table updated.")

# Chatbot interaction
def chatbot_interaction():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        response = generate_ai_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    training_data = ["How do I reset my password?", "What is your refund policy?", "How can I contact support?"]
    
    print("Training chatbot with reinforcement learning...")
    train_chatbot(training_data, episodes=50)

    print("Chatbot ready! Start chatting (type 'exit' to quit).")
    chatbot_interaction()
