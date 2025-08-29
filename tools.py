# tools.py
import json
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_books(path: str = "book_summaries.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_summary_by_title(title: str, path: str = "book_summaries_large.json") -> str:
    """
    Optional helper (can be used anywhere): returns long summary for a title.
    """
    books = _load_books(path)
    for b in books:
        if b.get("title","").lower() == title.lower():
            return b.get("summary", "")
    return ""
