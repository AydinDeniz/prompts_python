
import openai
import os

# Load OpenAI API key
API_KEY = "your_openai_api_key"

# Read code from file
def read_code(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Get AI-powered review from GPT
def review_code(code):
    prompt = f"Review the following Python code for best practices, security vulnerabilities, and optimizations:

{code}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a senior software engineer reviewing Python code."},
                  {"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

# Save review output
def save_review(output_text, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)

if __name__ == "__main__":
    code_file = "sample_code.py"  # Replace with actual file path
    output_file = "code_review.txt"

    print("Reading code...")
    code = read_code(code_file)

    print("Generating AI-powered review...")
    review = review_code(code)

    print("Saving review output...")
    save_review(review, output_file)

    print(f"Review saved to {output_file}")
