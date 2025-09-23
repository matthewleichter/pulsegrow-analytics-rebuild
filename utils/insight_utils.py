import re
from typing import List

def clean_text(text: str) -> str:
    """
    Cleans raw text by removing HTML tags, excessive whitespace, and common noise patterns.
    """
    text = re.sub(r'<[^>]+>', '', text)            # remove HTML tags
    text = re.sub(r'\s+', ' ', text)               # collapse multiple spaces
    text = re.sub(r'(http\S+)', '', text)          # remove URLs
    return text.strip()


def extract_key_sentences(text: str, n: int = 3) -> List[str]:
    """
    Extracts top n sentences that are likely to contain important information.
    This is a naive implementation based on sentence length and position.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) <= n:
        return sentences
    # prioritize first, last, and longest sentence
    sorted_sentences = sorted(sentences, key=len, reverse=True)
    return list({sentences[0], sorted_sentences[0], sentences[-1]})[:n]


def summarize_insights(text: str) -> str:
    """
    Produces a human-readable summary from raw LLM or report text.
    This can later be replaced with a GPT/LangChain-based summary.
    """
    if not text or len(text) < 50:
        return "Summary: Text too short for summarization."

    cleaned = clean_text(text)
    highlights = extract_key_sentences(cleaned)
    summary = "Summary: " + " ".join(highlights)
    return summary
