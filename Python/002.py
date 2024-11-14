import requests
from bs4 import BeautifulSoup
import csv
import time

# Base URL of the website (replace 'yourwebsite.com' with the actual URL)
base_url = 'https://www.yourwebsite.com/products/page/'

# Headers for the request to mimic a browser visit
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# List to store the product data
product_data = []


# Function to parse each page
def parse_page(page_num):
    url = f"{base_url}{page_num}"
    response = requests.get(url, headers=headers)

    # Check if the page request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve page {page_num}")
        return False

    soup = BeautifulSoup(response.text, 'html.parser')

    # Adjust selectors based on the website's structure
    products = soup.find_all('div', class_='product-item')  # Update with the correct tag and class

    if not products:
        return False  # Stop if no products are found on the page

    for product in products:
        # Extract product details
        name = product.find('h2', class_='product-name').get_text(strip=True)  # Adjust selector as needed
        price = product.find('span', class_='price').get_text(strip=True)  # Adjust selector as needed
        rating = product.find('span', class_='rating').get_text(strip=True) if product.find('span',
                                                                                            class_='rating') else "No rating"  # Adjust as needed
        product_url = product.find('a', class_='product-link')['href']  # Adjust selector as needed

        # Append data to the list
        product_data.append({
            'Name': name,
            'Price': price,
            'Rating': rating,
            'URL': product_url
        })

    return True


# Scrape multiple pages (change range as needed)
for page_num in range(1, 11):  # Adjust the range to cover the desired number of pages
    success = parse_page(page_num)
    if not success:
        print("No more products found, stopping.")
        break
    time.sleep(2)  # Be courteous to the server by adding a delay

# Write data to a CSV file
with open('products.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=['Name', 'Price', 'Rating', 'URL'])
    writer.writeheader()
    writer.writerows(product_data)

print("Data saved to products.csv")
