import argparse
import asyncio
import json
from pathlib import Path

from ollama import AsyncClient
from qdrant_client import AsyncQdrantClient

from app.settings import settings

KS = (1, 3, 5, 10)


def is_hit(payload: dict, source: str, sections: list[str]) -> bool:
    return payload["source"] == source and any(s in payload["section"] for s in sections)


async def main(collection: str, model: str, prefix: str, path: str) -> None:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    assert not any("TODO" in (r["source"], r["section"]) for r in rows), "остались TODO"

    qdrant = AsyncQdrantClient(settings.qdrant_host, port=settings.qdrant_port, check_compatibility=False)
    ollama = AsyncClient(host=settings.ollama_host)

    ranks, top_answerable, top_unanswerable = [], [], []
    for r in rows:
        emb = await ollama.embed(model=model, input=prefix + r["question"])
        res = await qdrant.query_points(collection, query=emb["embeddings"][0],
                                        limit=max(KS), with_payload=True)
        points = res.points
        if r["source"] is None:
            top_unanswerable.append(points[0].score)
            continue

        sections = r["section"] if isinstance(r["section"], list) else [r["section"]]
        rank = next((i for i, p in enumerate(points, 1)
                     if is_hit(p.payload, r["source"], sections)), None)
        ranks.append(rank)
        top_answerable.append(points[0].score)
        print(f"{r['id']} rank={rank}  {r['question']}")
        if rank is None or rank > 3:
            print("   получено:", [p.payload["section"][-45:] for p in points[:3]])

    n = len(ranks)
    print(f"\ncollection={collection} model={model} вопросов с ответом: {n}")
    for k in KS:
        print(f"recall@{k} = {sum(1 for x in ranks if x and x <= k)}/{n}")
    print(f"MRR = {sum(1 / x for x in ranks if x) / n:.3f}")
    if top_unanswerable:
        print(f"средний top-1 score: с ответом {sum(top_answerable) / n:.3f}, "
              f"без ответа {sum(top_unanswerable) / len(top_unanswerable):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=settings.collection_name)
    ap.add_argument("--model", default=settings.embedding_model)
    ap.add_argument("--prefix", default="", help="префикс запроса, например 'search_query: ' для nomic")
    ap.add_argument("--data", default="eval/rag_questions.jsonl")
    args = ap.parse_args()
    asyncio.run(main(args.collection, args.model, args.prefix, args.data))