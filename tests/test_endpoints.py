import pytest
from unittest.mock import patch

def test_query_question(client):
    '''Тест запроса к llm'''
    response = client.post('/query', json={'user_id': 1, 'query': 'Что такое RAG?'})
    assert response.status_code == 200

    data = response.json()
    assert 'response' in data

    content = data['response'].get('content').lower() if isinstance(data['response'], dict) else data['response']
    assert content
    assert 'rag' in content or 'retrieval-augmented generation' in content