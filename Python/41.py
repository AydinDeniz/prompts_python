
import ast
import openai

# Load OpenAI API key
API_KEY = "your_openai_api_key"

# Read Python code from file
def read_code(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Analyze code structure using AST
def analyze_code(code):
    try:
        tree = ast.parse(code)
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        return {"functions": functions, "lines": len(code.splitlines())}
    except SyntaxError:
        return {"error": "Invalid Python syntax"}

# Generate AI-powered bug fixes and optimizations
def fix_code_with_gpt(code):
    prompt = f"Review the following Python code and suggest fixes for bugs, optimizations, and security improvements:

{code}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a senior software engineer improving Python code."},
                  {"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

# Save improved code
def save_improved_code(output_text, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)

if __name__ == "__main__":
    code_file = "sample_code.py"  # Replace with actual file path
    output_file = "improved_code.py"

    print("Reading code...")
    code = read_code(code_file)

    print("Analyzing code structure...")
    analysis = analyze_code(code)
    print("Code Analysis:", analysis)

    print("Generating AI-powered improvements...")
    improved_code = fix_code_with_gpt(code)

    print("Saving improved code...")
    save_improved_code(improved_code, output_file)

    print(f"Improved code saved to {output_file}")
