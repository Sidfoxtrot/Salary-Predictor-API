import requests

# URL
url = 'http://localhost:5000/predict'

# Change the value of experience that you want to test
r = requests.post(url,json={'exp':4.9,})
print(r.json())
