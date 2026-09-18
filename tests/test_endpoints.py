import pytest
from unittest.mock import patch

def test_query_question(client):
    '''Тест вопроса к llm'''
    response = client.post('/query', json={'user_id': 1, 'query': 'Что такое RAG?'})
    assert response.status_code == 200

    data = response.json()
    assert 'response' in data

    content = data['response'].get('content').lower() if isinstance(data['response'], dict) else data['response']
    assert content
    assert 'rag' in content or 'retrieval-augmented generation' in content

def test_query_create_task(client):
    '''Тест создания задания'''
    '''Тест вопроса к llm'''
    response = client.post('/query', json={'user_id': 1, 'query': 'Создай новое задание'})
    assert response.status_code == 200

    data = response.json()
    assert 'response' in data

    content = data['response'].get('content').lower() if isinstance(data['response'], dict) else data['response']
    assert content

def test_query_add_comment(client):
    '''Тест добавления комментария'''
    response = client.post('/query', json={'user_id': 1, 'query': 'Добавь комментарий "нужно сделать" к заданию 1'})
    assert response.status_code == 200

    data = response.json()
    assert 'response' in data

    content = data['response'].get('content').lower() if isinstance(data['response'], dict) else data['response']
    assert content