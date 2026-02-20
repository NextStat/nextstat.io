"""Shared parsing utilities for benchmark scripts."""
import json


def parse_json_stdout(text):
    """Extract the first JSON object from CLI stdout, skipping any warn/log lines before it."""
    idx = text.find("{")
    if idx < 0:
        raise ValueError(f"No JSON object found in stdout:\n{text[:500]}")
    return json.loads(text[idx:])
