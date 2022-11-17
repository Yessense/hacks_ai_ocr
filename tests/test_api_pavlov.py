import requests

text = {
  "x": [
    "праблема"
  ]
}
bounds = requests.post("http://127.0.0.1:8081/model", json=text)

print(bounds.json())


