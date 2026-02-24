import hashlib


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="strict"))
    return h.hexdigest()

