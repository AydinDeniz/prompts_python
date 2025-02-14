
import scapy.all as scapy
import requests
import openai
import socket
import re
from bs4 import BeautifulSoup

# Load OpenAI API key
API_KEY = "your_openai_api_key"

# Perform network scan
def network_scan(target_ip):
    print(f"Scanning network: {target_ip}")
    arp_request = scapy.ARP(pdst=target_ip)
    broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
    arp_request_broadcast = broadcast / arp_request
    answered_list = scapy.srp(arp_request_broadcast, timeout=1, verbose=False)[0]

    devices = []
    for element in answered_list:
        devices.append({"ip": element[1].psrc, "mac": element[1].hwsrc})
    return devices

# Perform basic port scan
def port_scan(ip):
    open_ports = []
    for port in range(1, 1025):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((ip, port))
            if result == 0:
                open_ports.append(port)
            sock.close()
        except:
            continue
    return open_ports

# Perform SQL Injection detection
def detect_sql_injection(url):
    test_payloads = ["'", """, " OR 1=1 --", "' OR 'a'='a", "admin' --"]
    for payload in test_payloads:
        target_url = f"{url}?id={payload}"
        response = requests.get(target_url)
        if "error" in response.text.lower() or "sql" in response.text.lower():
            return True
    return False

# Perform XSS detection
def detect_xss(url):
    test_payloads = ["<script>alert('XSS')</script>", "" onmouseover="alert(1)"]
    for payload in test_payloads:
        target_url = f"{url}?q={payload}"
        response = requests.get(target_url)
        if payload in response.text:
            return True
    return False

# Perform Security Review using GPT
def gpt_security_review(code):
    prompt = f"Analyze the following Python code for security vulnerabilities and suggest improvements:

{code}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a cybersecurity expert analyzing Python code."},
                  {"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    target_ip = "192.168.1.1/24"
    target_url = "http://example.com"

    print("Performing network scan...")
    devices = network_scan(target_ip)
    print("Devices found:", devices)

    print("Performing port scan...")
    for device in devices:
        open_ports = port_scan(device["ip"])
        print(f"Open ports for {device['ip']}: {open_ports}")

    print("Checking for SQL Injection vulnerability...")
    if detect_sql_injection(target_url):
        print("SQL Injection vulnerability detected!")

    print("Checking for XSS vulnerability...")
    if detect_xss(target_url):
        print("XSS vulnerability detected!")

    print("Performing AI-driven security review...")
    sample_code = "def insecure_function():\n    os.system('rm -rf /')"
    security_review = gpt_security_review(sample_code)
    print("Security Review Results:\n", security_review)
