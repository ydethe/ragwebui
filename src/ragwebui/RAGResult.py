from dataclasses import dataclass


@dataclass
class RAGResult:
    source: str
    chunk_index: int
    ocr_used: bool
    text: str
