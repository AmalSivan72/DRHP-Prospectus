import requests

BASE_URL = "http://localhost:5000"   # adjust if your Flask app runs elsewhere
base = "pro_spdrdowjonesindustrialaverageetf"

# 1. GET all answers that need review (score < 75)
resp = requests.get(f"{BASE_URL}/review/{base}")
print("GET /review:", resp.status_code)
print(resp.json())

# 2. POST approve with edited answer (only works if score < 75)
payload = {
    "field": "Accounting Basis",   # must be a field with score < 75
    "action": "approve",
    "edited_answer": "PricewaterhouseCoopers LLP"
}
resp = requests.post(f"{BASE_URL}/review/{base}", json=payload)
print("POST approve:", resp.status_code)
print(resp.json())

# 3. POST reject (only works if score < 75)
payload = {
    "field": "NAV Calculation Method",   # must be a field with score < 75
    "action": "reject"
}
resp = requests.post(f"{BASE_URL}/review/{base}", json=payload)
print("POST reject:", resp.status_code)
print(resp.json())
