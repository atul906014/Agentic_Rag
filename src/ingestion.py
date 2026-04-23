from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


def read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def load_documents(data_dir: Path, chunk_size: int, chunk_overlap: int) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue

        text = read_file(path)
        for index, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            documents.append(
                {
                    "id": f"{path.name}-{index}",
                    "source": str(path),
                    "content": chunk,
                }
            )
    return documents
