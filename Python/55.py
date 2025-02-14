
import requests
import shodan
import whois
import socket
import json
from bs4 import BeautifulSoup
import openai

# Load OpenAI API key
API_KEY = "your_openai_api_key"

# Load Shodan API key
SHODAN_API_KEY = "your_shodan_api_key"
shodan_api = shodan.Shodan(SHODAN_API_KEY)

# Perform WHOIS lookup
def whois_lookup(domain):
    print(f"Performing WHOIS lookup on {domain}...")
    whois_info = whois.whois(domain)
    return whois_info

# Perform Shodan scan
def shodan_scan(ip):
    print(f"Scanning {ip} with Shodan...")
    try:
        result = shodan_api.host(ip)
        return result
    except shodan.APIError as e:
        return {"error": str(e)}

# Scrape website metadata
def scrape_metadata(url):
    print(f"Scraping metadata from {url}...")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    metadata = {
        "title": soup.title.string if soup.title else "N/A",
        "description": soup.find("meta", attrs={"name": "description"})["content"] if soup.find("meta", attrs={"name": "description"}) else "N/A",
        "keywords": soup.find("meta", attrs={"name": "keywords"})["content"] if soup.find("meta", attrs={"name": "keywords"}) else "N/A"
    }
    return metadata

# Generate OSINT report using GPT-4
def generate_osint_report(target, whois_data, shodan_data, metadata):
    print("Generating AI-powered OSINT report...")
    report_data = {
        "target": target,
        "whois": str(whois_data),
        "shodan": json.dumps(shodan_data, indent=4),
        "metadata": metadata
    }

    prompt = f"Analyze the following OSINT reconnaissance data for potential security risks:
{json.dumps(report_data, indent=4)}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a cybersecurity OSINT analyst."},
                  {"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    target_domain = "example.com"
    target_ip = socket.gethostbyname(target_domain)

    whois_data = whois_lookup(target_domain)
    shodan_data = shodan_scan(target_ip)
    metadata = scrape_metadata(f"http://{target_domain}")

    osint_report = generate_osint_report(target_domain, whois_data, shodan_data, metadata)

    with open("osint_report.txt", "w", encoding="utf-8") as f:
        f.write(osint_report)

    print("OSINT report saved to 'osint_report.txt'.")
