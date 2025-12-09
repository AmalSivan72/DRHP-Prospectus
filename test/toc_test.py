import requests

def test_get_toc(base: str):
    url = f"http://localhost:5000/get_toc/{base}"
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        try:
            data = response.json()
            print("Response JSON:")
            print(data)
        except Exception as e:
            print("❌ Failed to parse JSON response")
            print("Raw Response:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    # Replace with your actual base name (without .pdf)
    base_name = "lgeindiadrhp_20241206142942"
    test_get_toc(base_name)
