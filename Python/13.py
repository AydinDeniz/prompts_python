
import threading
import queue
import time
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import json
from urllib.robotparser import RobotFileParser

class DistributedWebScraper:
    def __init__(self, urls, output_file, delay=1):
        self.urls = queue.Queue()
        for url in urls:
            self.urls.put(url)
        self.output_file = output_file
        self.delay = delay
        self.lock = threading.Lock()
        self.scraped_data = []
        self.robot_parsers = {}

    def is_allowed(self, url):
        domain = "/".join(url.split("/")[:3]) + "/robots.txt"
        if domain not in self.robot_parsers:
            rp = RobotFileParser()
            rp.set_url(domain)
            rp.read()
            self.robot_parsers[domain] = rp
        return self.robot_parsers[domain].can_fetch("*", url)

    def scrape(self):
        while not self.urls.empty():
            url = self.urls.get()
            if not self.is_allowed(url):
                print(f"Blocked by robots.txt: {url}")
                continue
            try:
                response = requests.get(url, timeout=5)
                time.sleep(self.delay)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    title = soup.title.string if soup.title else "No Title"
                    body = soup.get_text()
                    data = {"url": url, "title": title, "content": body}
                    with self.lock:
                        self.scraped_data.append(data)
                        print(f"Scraped: {url}")
            except Exception as e:
                print(f"Error scraping {url}: {e}")

    def save_data(self):
        with open(self.output_file, "w", encoding="utf-8") as file:
            json.dump(self.scraped_data, file, ensure_ascii=False, indent=4)
        print(f"Data saved to {self.output_file}")

    def start(self, num_threads=5):
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=self.scrape)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        self.save_data()

if __name__ == "__main__":
    urls_to_scrape = [
        "https://example.com",
        "https://example.org",
        "https://example.net"
    ]
    scraper = DistributedWebScraper(urls=urls_to_scrape, output_file="scraped_data.json", delay=2)
    scraper.start(num_threads=3)
