
import time
import csv
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 2Captcha API key
API_KEY = "your_2captcha_api_key"

# Function to solve CAPTCHA using 2Captcha
def solve_captcha(site_key, url):
    response = requests.post("http://2captcha.com/in.php", {
        "key": API_KEY,
        "method": "userrecaptcha",
        "googlekey": site_key,
        "pageurl": url,
        "json": 1
    })
    request_id = response.json().get("request")
    
    if not request_id:
        print("Error requesting CAPTCHA solve.")
        return None

    # Wait for the CAPTCHA to be solved
    time.sleep(15)
    
    for _ in range(20):
        result = requests.get(f"http://2captcha.com/res.php?key={API_KEY}&action=get&id={request_id}&json=1")
        if result.json().get("status") == 1:
            return result.json().get("request")
        time.sleep(5)

    print("CAPTCHA solving failed.")
    return None

# Initialize Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Target website (replace with actual URL)
url = "https://example.com"

driver.get(url)
time.sleep(5)  # Wait for page to load

# Extract reCAPTCHA site-key (modify selector based on the site)
site_key_element = driver.find_element(By.XPATH, "//div[@class='g-recaptcha']")
site_key = site_key_element.get_attribute("data-sitekey")

# Solve CAPTCHA
captcha_solution = solve_captcha(site_key, url)

if captcha_solution:
    # Inject CAPTCHA solution into the site
    driver.execute_script(f"document.getElementById('g-recaptcha-response').innerHTML = '{captcha_solution}';")
    driver.find_element(By.XPATH, "//input[@type='submit']").click()
    time.sleep(5)

# Extract data (modify based on actual site structure)
data = []
elements = driver.find_elements(By.CLASS_NAME, "data-element")
for el in elements:
    data.append(el.text)

# Save to CSV
csv_filename = "scraped_data.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Extracted Data"])
    writer.writerows([[d] for d in data])

print(f"Data saved to {csv_filename}")
driver.quit()
