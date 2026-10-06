import json
import sys
from langchain_core.messages import HumanMessage
from app.agent import call_model
from app.logger import logger

TOOL_TO_ACTION = {"answer": "answer_rag", "create_task": "create_task", "add_comment": "add_comment"}
LABELS = ["answer_rag", "create_task", "add_comment", "none"]

split = sys.argv[1] if len(sys.argv) > 1 else None
rows = [json.loads(line) for line in open("/main/eval/routing.jsonl", encoding="utf-8") if line.strip()]
rows = [r for r in rows if split is None or r["split"] == split]

pairs = []  # (expected, predicted)
listt = ['r039', 'r041', 'r043', 'r045', 'r047', 'r049'] #debug
for r in rows:
    if r['id'] not in listt: #debug
        continue
    state = {"messages": [HumanMessage(content=r["query"])], "task_id": r["state_task_id"]}
    msg = call_model(state)["messages"][0]
    calls = msg.tool_calls
    pred = TOOL_TO_ACTION.get(calls[0]["name"], "other") if calls else "none"
    pairs.append((r["expected_action"], pred))
    if pred != r["expected_action"]:
        print(f"{r['id']} {r['query']!r}: expected={r['expected_action']} got={f'{pred}, {msg.content}'}")
        logger.info(f"got={pred}, text={msg.content!r}, calls={calls}")

for lab in LABELS:
    tp = sum(e == lab and p == lab for e, p in pairs)
    n_exp = sum(e == lab for e, _ in pairs)
    n_pred = sum(p == lab for _, p in pairs)
    print(f"{lab:12} precision={tp / n_pred if n_pred else 0:.2f}  recall={tp / n_exp if n_exp else 0:.2f}")