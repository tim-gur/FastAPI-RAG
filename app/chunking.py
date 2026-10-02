import uuid
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def load_chunks(docs_dir: str = "docs", chunk_size: int = 1000, overlap: int = 150) -> list[dict]:

    # Инициализация сплиттеров
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )

    chunks = []
    for path in sorted(Path(docs_dir).rglob("*.md")):
        source = str(path.relative_to(docs_dir))
        sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
        for i, doc in enumerate(char_splitter.split_documents(sections)):
            section = " > ".join(
                doc.metadata[k] for k in ("h1", "h2", "h3") if k in doc.metadata
            )
            chunks.append({
                "id": str(uuid.uuid5(NAMESPACE, f"{source}:{i}")),
                "text": doc.page_content,
                "source": source,
                "section": section,
            })
    return chunks