import requests

BASE_URL = "http://127.0.0.1:5000"   # adjust if your Flask app runs elsewhere
BASE_NAME = "pro_spdrdowjonesindustrialaverageetf"                 # replace with the actual base you used

def get_status_details(base):
    url = f"{BASE_URL}/status/{base}/details"
    resp = requests.get(url)
    if resp.status_code == 200:
        print("\n--- STATUS DETAILS ---")
        print(resp.json())
    else:
        print(f"Error {resp.status_code}: {resp.text}")

def get_completed_answers(base):
    url = f"{BASE_URL}/status/{base}/answers"
    resp = requests.get(url)
    if resp.status_code == 200:
        print("\n--- COMPLETED ANSWERS ---")
        print(resp.json())
    else:
        print(f"Error {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    # Run both endpoints
    get_status_details(BASE_NAME)
    get_completed_answers(BASE_NAME)
