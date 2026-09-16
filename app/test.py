import requests

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
params = {'user_id': 123}
json = {'query': 'добавь комментарий к этой задаче: сделано'}
response = requests.post('http://127.0.0.1:8000' + '/ask', headers=headers, params=params, json=json)
print(response)
print(response.json()['response'])