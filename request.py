import argparse
import json

import requests

# Set up the argument parser
parser = argparse.ArgumentParser(
    description="Send a text to the server for processing."
)
parser.add_argument("text", type=str, help="The text to send to the server")
parser.add_argument(
    "-n", "--num-tokens", type=int, default=128, help="The number of tokens to generate"
)
parser.add_argument(
    "-t", "--temp", type=float, default=0.75, help="The temperature for the generation"
)

# Parse the arguments
args = parser.parse_args()

url = "http://localhost:5000/predict"
data = {"text": args.text, "tokens": 128, "temp": 0.75}
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)

print(response.json())
