from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    hf_token: str
    chat_model: str
    embedding_model: str
    data_dir: Path
    index_dir: Path
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4


def get_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    return Settings(
        hf_token=os.getenv("HF_TOKEN", ""),
        chat_model=os.getenv("HF_CHAT_MODEL", "HuggingFaceH4/zephyr-7b-beta"),
        embedding_model=os.getenv(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        data_dir=root / "data",
        index_dir=root / "storage",
    )
