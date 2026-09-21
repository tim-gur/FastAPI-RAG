import requests
from app.logger import logger

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

json = {'user_id': 123, 'query': 'что такое RAG?'}
resp_quest = (requests.post('http://127.0.0.1:8000' + '/query', headers=headers, json=json)).json()

json = {'user_id': 123, 'query': 'создай новую задачу?'}
resp_task = (requests.post('http://127.0.0.1:8000' + '/query', headers=headers, json=json)).json()

json = {'user_id': 123, 'query': 'добавь комментарий к этой задаче: сделано'}
resp_comm = (requests.post('http://127.0.0.1:8000' + '/query', headers=headers, json=json)).json()

logger.info(f'Question request: {resp_quest}')
logger.info(f'Task request: {resp_task}')
logger.info(f'Comment request: {resp_comm}')