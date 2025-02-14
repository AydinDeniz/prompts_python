
from celery import Celery
import requests
import json

# Initialize Celery
app = Celery("tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

# Fetch real-time stock prices
@app.task
def fetch_stock_price(symbol):
    api_url = f"https://api.example.com/stocks/{symbol}"  # Replace with actual API
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    return {"error": "Failed to fetch data"}

# Process stock data
@app.task
def process_stock_data(stock_data):
    try:
        prices = [entry["price"] for entry in stock_data["data"]]
        avg_price = sum(prices) / len(prices)
        return {"symbol": stock_data["symbol"], "average_price": avg_price}
    except Exception as e:
        return {"error": str(e)}

# Save results to a file
@app.task
def save_results(result, file_path="stock_results.json"):
    with open(file_path, "w") as f:
        json.dump(result, f, indent=4)
    return f"Results saved to {file_path}"

if __name__ == "__main__":
    symbol = "AAPL"
    stock_data = fetch_stock_price.delay(symbol)
    processed_data = process_stock_data.delay(stock_data.get())
    save_results.delay(processed_data.get())
