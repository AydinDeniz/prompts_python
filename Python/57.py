
import subprocess
import openai
import os
import json
from capstone import Cs, CS_ARCH_X86, CS_MODE_64
import lief

# Load OpenAI API key
API_KEY = "your_openai_api_key"

# Disassemble binary using Capstone
def disassemble_binary(binary_path):
    print("Disassembling binary...")
    binary = lief.parse(binary_path)
    code_section = next((s for s in binary.sections if s.name == ".text"), None)

    if not code_section:
        print("Error: No executable code found in binary.")
        return []

    md = Cs(CS_ARCH_X86, CS_MODE_64)
    instructions = []

    for instr in md.disasm(code_section.content, binary.entrypoint):
        instructions.append(f"0x{instr.address:x}:	{instr.mnemonic}	{instr.op_str}")

    return instructions

# Extract strings from binary
def extract_strings(binary_path):
    print("Extracting strings from binary...")
    result = subprocess.run(["strings", binary_path], capture_output=True, text=True)
    return result.stdout.split("\n")

# Analyze binary with GPT-4
def analyze_binary_gpt(instructions, strings):
    print("Analyzing binary with GPT-4...")
    prompt = (
        "Given the following assembly instructions and extracted strings, analyze the binary file for security risks, "
        "obfuscation techniques, and potential functionality.

"
        "Assembly Instructions:
"
        f"{json.dumps(instructions[:50], indent=4)}

"
        "Extracted Strings:
"
        f"{json.dumps(strings[:20], indent=4)}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an expert in reverse engineering and binary analysis."},
                  {"role": "user", "content": prompt}],
        max_tokens=800
    )
    return response["choices"][0]["message"]["content"]

# Extract metadata from binary
def extract_binary_metadata(binary_path):
    print("Extracting binary metadata...")
    binary = lief.parse(binary_path)

    metadata = {
        "name": binary.name,
        "entrypoint": hex(binary.entrypoint),
        "architecture": binary.format.name,
        "libraries": binary.libraries
    }
    return metadata

# Save analysis report
def save_report(report):
    with open("reverse_engineering_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("Reverse engineering report saved to 'reverse_engineering_report.txt'.")

if __name__ == "__main__":
    binary_file = "sample_binary"

    if not os.path.exists(binary_file):
        print("Error: Binary file not found!")
        exit()

    instructions = disassemble_binary(binary_file)
    strings = extract_strings(binary_file)
    metadata = extract_binary_metadata(binary_file)
    
    print("Generating AI-powered analysis report...")
    analysis_report = analyze_binary_gpt(instructions, strings)

    full_report = f"Binary Metadata:\n{json.dumps(metadata, indent=4)}\n\n{analysis_report}"
    save_report(full_report)
