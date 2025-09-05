import requests
import json
init_req = {
           "session_id": "TestId",
            "user_input": "Perdón, me refería a si me podrías decir qué pone en la carta.",
            "img": True,
            "img_base64": None,
}

response = requests.post(url='http://127.0.0.1:8000/orchestrate', json=init_req, verify=False)
print(response)
try:
    json_response = json.loads(response.text)
    print(json_response)
except Exception as e:
    print("Error parsing response as JSON:", str(e))
    print("Response text:", response.text)