
import requests
import json
import openai
import datetime
import os
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load OpenAI API key
API_KEY = "your_openai_api_key"

# Load threat intelligence sources
THREAT_FEEDS = [
    "https://otx.alienvault.com/api/v1/pulses/subscribed",
    "https://www.spamhaus.org/drop/drop.txt",
    "https://www.abuseipdb.com/sitemap",
]

# Scrape threat feeds
def fetch_threat_data():
    print("Fetching cyber threat intelligence data...")
    threat_data = []
    
    for url in THREAT_FEEDS:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.text
                threat_data.append({"source": url, "data": data})
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch data from {url}: {e}")

    return threat_data

# Extract IoCs (Indicators of Compromise) from raw data
def extract_iocs(threat_data):
    print("Extracting Indicators of Compromise (IoCs)...")
    iocs = []

    for feed in threat_data:
        for line in feed["data"].split("\n"):
            if "." in line and not line.startswith("#"):  # Simple IP/domain filter
                iocs.append(line.strip())

    return list(set(iocs))

# Enrich IoCs using VirusTotal API
def enrich_iocs(iocs):
    print("Enriching IoCs using VirusTotal...")
    enriched_iocs = []

    for ioc in iocs[:10]:  # Limit API calls
        response = requests.get(f"https://www.virustotal.com/api/v3/ip_addresses/{ioc}",
                                headers={"x-apikey": "your_virustotal_api_key"})
        if response.status_code == 200:
            enriched_data = response.json()
            enriched_iocs.append({"ioc": ioc, "details": enriched_data})

    return enriched_iocs

# Perform NLP-based threat classification
def classify_threats(iocs):
    print("Classifying threats using AI...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    categories = ["Malware", "Phishing", "Botnet", "Ransomware", "DDoS", "Trojan", "Spyware"]
    classified_iocs = []

    for ioc in iocs[:10]:  # Process limited IoCs
        result = classifier(ioc, candidate_labels=categories)
        classified_iocs.append({"ioc": ioc, "classification": result["labels"][0], "score": result["scores"][0]})

    return classified_iocs

# Cluster IoCs using machine learning
def cluster_iocs(iocs):
    print("Clustering IoCs using K-Means...")
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(iocs)

    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    clustered_data = [{"ioc": iocs[i], "cluster": int(kmeans.labels_[i])} for i in range(len(iocs))]

    plt.scatter(range(len(iocs)), kmeans.labels_, c=kmeans.labels_, cmap="viridis")
    plt.title("Threat Intelligence Clustering")
    plt.xlabel("IoC Index")
    plt.ylabel("Cluster")
    plt.savefig("threat_clusters.png")
    plt.close()

    return clustered_data

# Generate AI-driven threat analysis report
def generate_threat_report(iocs, classified_iocs, clusters):
    print("Generating AI-powered threat intelligence report...")
    
    prompt = (
        "Analyze the following cyber threat intelligence data, including Indicators of Compromise (IoCs), "
        "threat classifications, and cluster insights. Provide an executive summary of the potential cybersecurity risks.

"
        f"IoCs: {json.dumps(iocs[:10], indent=4)}

"
        f"Classified Threats: {json.dumps(classified_iocs, indent=4)}

"
        f"Cluster Insights: {json.dumps(clusters[:10], indent=4)}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a cybersecurity expert analyzing threat intelligence data."},
                  {"role": "user", "content": prompt}],
        max_tokens=1000
    )
    
    report = response["choices"][0]["message"]["content"]
    
    with open("threat_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    return report

if __name__ == "__main__":
    threat_data = fetch_threat_data()
    iocs = extract_iocs(threat_data)
    enriched_iocs = enrich_iocs(iocs)
    classified_iocs = classify_threats(iocs)
    clusters = cluster_iocs(iocs)
    
    threat_report = generate_threat_report(iocs, classified_iocs, clusters)
    
    print("Threat Intelligence Report Generated and Saved.")
