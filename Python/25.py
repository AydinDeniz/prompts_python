
import openai
import json

# Load API key from config file
CONFIG_FILE = "config.json"
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

API_KEY = config["openai_api_key"]

# OpenAI API call for text summarization
def summarize_text(text, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "Summarize the following text."},
                  {"role": "user", "content": text}],
        max_tokens=100
    )
    return response["choices"][0]["message"]["content"]

# Read input text file
INPUT_FILE = "input_text.txt"
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Summarize text
summary = summarize_text(text)

# Save summary to output file
OUTPUT_FILE = "summary.txt"
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(summary)

print(f"Summary saved to {OUTPUT_FILE}")
