from unittest.mock import patch
from app.agent import create_task, add_comment

def test_create_task_tool(temp_db):
    with patch("app.agent.create_task_sql", return_value=7):
        result = create_task.invoke({"user_id": 42})
    assert result == 7

def test_add_comment_tool_success():
    with patch("app.agent.add_comment_sql") as mock_add:
        result = add_comment.invoke({"user_id": 1, "task_id": 7, "comment": "новая задача"})
    mock_add.assert_called_once_with(1, 7, "новая задача")
    assert "7" in result