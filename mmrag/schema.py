from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class Document:
    doc_id: str              # "ai2d_0042_txt_0" or "chartqa_0015_img"
    group_id: str            # "ai2d_0042" or "chartqa_0015"
    doc_type: str            # "text" or "image"
    source_dataset: str      # "ai2d" or "chartqa"
    content: str             # For text: "Question: ... Answer: ..." / For image: ""
    image_path: str = ""     # For image: relative path / For text: ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(**d)


@dataclass
class RetrievalResult:
    doc_id: str
    score: float
    doc_type: str            # "text" or "image"
    content: str             # The text content or empty for images
    image_path: str = ""
    group_id: str = ""
    metadata: dict = field(default_factory=dict)


def save_documents(docs: list[Document], path: str) -> None:
    with open(path, "w") as f:
        json.dump([d.to_dict() for d in docs], f, indent=2)


def load_documents(path: str) -> list[Document]:
    with open(path, "r") as f:
        return [Document.from_dict(d) for d in json.load(f)]