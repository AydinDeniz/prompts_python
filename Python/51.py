
import tweepy
import openai
import random
import time
import json
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
from transformers import pipeline

# Load Twitter API credentials
TWITTER_CREDENTIALS = "twitter_credentials.json"
with open(TWITTER_CREDENTIALS, "r") as f:
    creds = json.load(f)

# Authenticate with Twitter API
auth = tweepy.OAuth1UserHandler(creds["API_KEY"], creds["API_SECRET_KEY"], creds["ACCESS_TOKEN"], creds["ACCESS_SECRET"])
api = tweepy.API(auth, wait_on_rate_limit=True)

# Load OpenAI API key
API_KEY = "your_openai_api_key"

# Sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Generate counter-narrative tweets
def generate_counter_narrative(tweet_text, sentiment):
    if sentiment == "POSITIVE":
        prompt = f"Write a controversial response to this positive tweet: {tweet_text}"
    else:
        prompt = f"Write an uplifting and positive counter-response to this negative tweet: {tweet_text}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a social media strategist specializing in sentiment influence."},
                  {"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response["choices"][0]["message"]["content"]

# Fetch recent tweets on a given topic
def fetch_tweets(topic, count=50):
    tweets = api.search_tweets(q=topic, lang="en", count=count, tweet_mode="extended")
    return [{"text": tweet.full_text, "id": tweet.id, "user": tweet.user.screen_name} for tweet in tweets]

# Analyze and manipulate sentiment
def sentiment_manipulation(topic):
    tweets = fetch_tweets(topic)
    manipulated_tweets = []

    for tweet in tweets:
        sentiment_result = sentiment_analyzer(tweet["text"])[0]
        counter_text = generate_counter_narrative(tweet["text"], sentiment_result["label"])
        
        manipulated_tweets.append({
            "original_tweet": tweet["text"],
            "sentiment": sentiment_result["label"],
            "counter_narrative": counter_text
        })

    return manipulated_tweets

# Simulate social network influence
def simulate_social_network_effect(tweets):
    G = nx.DiGraph()
    
    for i, tweet in enumerate(tweets):
        G.add_node(f"Tweet_{i}", sentiment=tweet["sentiment"])
        if i > 0:
            G.add_edge(f"Tweet_{i-1}", f"Tweet_{i}")

    pos = nx.spring_layout(G)
    node_colors = ["green" if G.nodes[node]["sentiment"] == "POSITIVE" else "red" for node in G.nodes]

    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=1000, font_size=8)
    plt.title("Social Media Sentiment Influence Simulation")
    plt.show()

if __name__ == "__main__":
    topic = "climate change"
    print(f"Analyzing and manipulating sentiment on topic: {topic}")
    manipulated_data = sentiment_manipulation(topic)

    with open("sentiment_manipulation_results.json", "w", encoding="utf-8") as f:
        json.dump(manipulated_data, f, indent=4)

    print("Simulating social network influence...")
    simulate_social_network_effect(manipulated_data)
