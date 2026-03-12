"""Intent Router — classifies user messages using TF-IDF cosine similarity.

Falls back to DEFAULT_INTENT when no category matches or config is empty.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from src.config.models import IntentRouterConfig

VALID_INTENTS = frozenset(
    {"chitchat", "faq_query", "customer_query", "document_upload", "document_query"}
)

DEFAULT_INTENT = "faq_query"


@dataclass
class IntentResult:
    intent: str
    confidence: float
    response: str | None = None


class IntentRouter:
    """Classifies user messages via TF-IDF cosine similarity against utterance examples."""

    def __init__(self, config: IntentRouterConfig) -> None:
        self._config = config
        self._category_vectors: dict[str, list[dict[str, float]]] = {}
        self._category_responses: dict[str, str | None] = {}
        self._category_actions: dict[str, str] = {}
        self._idf: dict[str, float] = {}
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _compute_tf(self, tokens: list[str]) -> dict[str, float]:
        tf: dict[str, float] = {}
        if not tokens:
            return tf
        for token in tokens:
            tf[token] = tf.get(token, 0.0) + 1.0
        for token in tf:
            tf[token] /= len(tokens)
        return tf

    def _build_index(self) -> None:
        all_docs: list[list[str]] = []
        category_utterance_tokens: dict[str, list[list[str]]] = {}

        for cat_name, cat_config in self._config.categories.items():
            self._category_responses[cat_name] = cat_config.response
            self._category_actions[cat_name] = cat_config.action
            tokens_list: list[list[str]] = []
            for utterance in cat_config.utterances:
                tokens = self._tokenize(utterance)
                if tokens:
                    tokens_list.append(tokens)
                    all_docs.append(tokens)
            category_utterance_tokens[cat_name] = tokens_list

        total_docs = len(all_docs)
        if total_docs == 0:
            return

        doc_freq: dict[str, int] = {}
        for doc_tokens in all_docs:
            for token in set(doc_tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        self._idf = {
            token: math.log((total_docs + 1) / (freq + 1)) + 1
            for token, freq in doc_freq.items()
        }

        for cat_name, tokens_list in category_utterance_tokens.items():
            vectors: list[dict[str, float]] = []
            for tokens in tokens_list:
                tf = self._compute_tf(tokens)
                tfidf = {t: v * self._idf.get(t, 1.0) for t, v in tf.items()}
                vectors.append(tfidf)
            self._category_vectors[cat_name] = vectors

    def _cosine_similarity(self, vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(vec_a[t] * vec_b.get(t, 0.0) for t in vec_a)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _message_to_tfidf(self, message: str) -> dict[str, float]:
        tokens = self._tokenize(message)
        tf = self._compute_tf(tokens)
        return {t: v * self._idf.get(t, 1.0) for t, v in tf.items()}

    def classify(self, message: str) -> IntentResult:
        if not message or not message.strip():
            return IntentResult(intent=DEFAULT_INTENT, confidence=0.0)

        if not self._category_vectors:
            return IntentResult(intent=DEFAULT_INTENT, confidence=0.0)

        message_vec = self._message_to_tfidf(message)
        best_intent = DEFAULT_INTENT
        best_score = 0.0

        for cat_name, vectors in self._category_vectors.items():
            for vec in vectors:
                score = self._cosine_similarity(message_vec, vec)
                if score > best_score:
                    best_score = score
                    best_intent = cat_name

        if best_intent not in VALID_INTENTS:
            best_intent = DEFAULT_INTENT

        response = None
        if best_intent == "chitchat" and self._category_responses.get("chitchat"):
            response = self._category_responses["chitchat"]

        return IntentResult(
            intent=best_intent,
            confidence=round(best_score, 4),
            response=response,
        )
