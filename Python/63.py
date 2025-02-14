
import angr
import openai
import os
import json
import subprocess

# Load OpenAI API key
API_KEY = "your_openai_api_key"

# Analyze binary for vulnerabilities using Angr
def analyze_binary(binary_path):
    print(f"Analyzing binary {binary_path} for vulnerabilities...")
    project = angr.Project(binary_path, auto_load_libs=False)

    cfg = project.analyses.CFGFast()
    functions = list(cfg.kb.functions.items())

    exploitable_functions = []
    for addr, func in functions:
        if "strcpy" in func.name or "gets" in func.name or "system" in func.name:
            exploitable_functions.append({"name": func.name, "address": hex(addr)})

    return exploitable_functions

# Generate AI-powered exploit payloads
def generate_exploit_payload(binary_path, vulnerabilities):
    print("Generating AI-powered exploit payloads...")
    prompt = (
        "Given the following binary file and detected vulnerabilities, generate possible exploit payloads.

"
        f"Binary: {binary_path}
"
        f"Vulnerabilities: {json.dumps(vulnerabilities, indent=4)}
"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a cybersecurity expert specializing in binary exploitation."},
                  {"role": "user", "content": prompt}],
        max_tokens=800
    )
    
    return response["choices"][0]["message"]["content"]

# Perform fuzz testing on the binary
def fuzz_test_binary(binary_path):
    print(f"Fuzz testing {binary_path} for unexpected behavior...")
    fuzz_results = []
    fuzz_inputs = [b"A" * i for i in range(10, 500, 50)]

    for fuzz_input in fuzz_inputs:
        try:
            result = subprocess.run([binary_path], input=fuzz_input, capture_output=True, timeout=1)
            crash_detected = result.returncode != 0
            fuzz_results.append({"input_size": len(fuzz_input), "crash": crash_detected})
        except subprocess.TimeoutExpired:
            fuzz_results.append({"input_size": len(fuzz_input), "crash": True})

    return fuzz_results

# Save exploit report
def save_report(report):
    with open("exploit_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Exploit report saved to 'exploit_report.txt'.")

if __name__ == "__main__":
    binary_file = "vulnerable_binary"

    if not os.path.exists(binary_file):
        print("Error: Binary file not found!")
        exit()

    vulnerabilities = analyze_binary(binary_file)
    fuzz_results = fuzz_test_binary(binary_file)

    print("Generating AI-assisted exploit report...")
    exploit_payloads = generate_exploit_payload(binary_file, vulnerabilities)

    full_report = (
        f"Vulnerability Analysis:\n{json.dumps(vulnerabilities, indent=4)}\n\n"
        f"Fuzz Testing Results:\n{json.dumps(fuzz_results, indent=4)}\n\n"
        f"AI-Generated Exploit Payloads:\n{exploit_payloads}"
    )
    save_report(full_report)
