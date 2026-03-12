"""Unit tests for IntentRouter."""

from __future__ import annotations

import pytest

from src.config.models import IntentCategory, IntentRouterConfig
from src.router.intent import DEFAULT_INTENT, VALID_INTENTS, IntentResult, IntentRouter


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def sample_config() -> IntentRouterConfig:
    """Örnek intent router yapılandırması."""
    return IntentRouterConfig(
        categories={
            "chitchat": IntentCategory(
                utterances=["merhaba", "nasılsın", "selam", "günaydın", "hello", "hi"],
                action="direct_response",
                response="Merhaba! Size nasıl yardımcı olabilirim?",
            ),
            "faq_query": IntentCategory(
                utterances=[
                    "sık sorulan sorular",
                    "nasıl yapılır",
                    "yardım",
                    "bilgi almak istiyorum",
                ],
                action="agent_loop",
                response=None,
            ),
            "customer_query": IntentCategory(
                utterances=[
                    "müşteri bilgisi",
                    "abonelik durumu",
                    "fatura sorgula",
                    "destek talebi",
                    "hesap bilgileri",
                ],
                action="agent_loop",
                response=None,
            ),
            "document_upload": IntentCategory(
                utterances=[
                    "doküman yükle",
                    "dosya yükle",
                    "PDF yükle",
                    "belge ekle",
                ],
                action="agent_loop",
                response=None,
            ),
            "document_query": IntentCategory(
                utterances=[
                    "dokümanda ara",
                    "raporda ne yazıyor",
                    "belgeye göre",
                    "doküman sorgula",
                ],
                action="agent_loop",
                response=None,
            ),
        }
    )


@pytest.fixture
def router(sample_config: IntentRouterConfig) -> IntentRouter:
    return IntentRouter(sample_config)


# ------------------------------------------------------------------
# IntentResult yapısı
# ------------------------------------------------------------------


class TestIntentResult:
    def test_result_has_required_fields(self) -> None:
        result = IntentResult(intent="chitchat", confidence=0.95, response="Hi")
        assert result.intent == "chitchat"
        assert result.confidence == 0.95
        assert result.response == "Hi"

    def test_result_response_defaults_to_none(self) -> None:
        result = IntentResult(intent="faq_query", confidence=0.5)
        assert result.response is None


# ------------------------------------------------------------------
# Sınıflandırma sonucu her zaman geçerli bir intent döndürür
# ------------------------------------------------------------------


class TestClassifyReturnsValidIntent:
    def test_result_intent_is_valid(self, router: IntentRouter) -> None:
        result = router.classify("merhaba")
        assert result.intent in VALID_INTENTS

    def test_confidence_between_zero_and_one(self, router: IntentRouter) -> None:
        result = router.classify("merhaba")
        assert 0.0 <= result.confidence <= 1.0

    def test_unknown_message_returns_valid_intent(self, router: IntentRouter) -> None:
        result = router.classify("xyzzy gibberish 12345")
        assert result.intent in VALID_INTENTS


# ------------------------------------------------------------------
# Örnek ifade eşleştirme — utterance'lar doğru kategoriye sınıflandırılmalı
# ------------------------------------------------------------------


class TestUtteranceMatching:
    def test_chitchat_utterance(self, router: IntentRouter) -> None:
        result = router.classify("merhaba")
        assert result.intent == "chitchat"

    def test_hello_utterance(self, router: IntentRouter) -> None:
        result = router.classify("hello")
        assert result.intent == "chitchat"

    def test_faq_utterance(self, router: IntentRouter) -> None:
        result = router.classify("sık sorulan sorular")
        assert result.intent == "faq_query"

    def test_customer_query_utterance(self, router: IntentRouter) -> None:
        result = router.classify("fatura sorgula")
        assert result.intent == "customer_query"

    def test_document_upload_utterance(self, router: IntentRouter) -> None:
        result = router.classify("doküman yükle")
        assert result.intent == "document_upload"

    def test_document_query_utterance(self, router: IntentRouter) -> None:
        result = router.classify("dokümanda ara")
        assert result.intent == "document_query"


# ------------------------------------------------------------------
# Chitchat — LLM çağrısı yapmadan önceden tanımlı yanıt
# ------------------------------------------------------------------


class TestChitchatDirectResponse:
    def test_chitchat_returns_predefined_response(self, router: IntentRouter) -> None:
        result = router.classify("selam")
        assert result.intent == "chitchat"
        assert result.response is not None
        assert "yardımcı" in result.response.lower()

    def test_non_chitchat_has_no_response(self, router: IntentRouter) -> None:
        result = router.classify("fatura sorgula")
        assert result.response is None


# ------------------------------------------------------------------
# Boş / edge-case girdiler
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string(self, router: IntentRouter) -> None:
        result = router.classify("")
        assert result.intent in VALID_INTENTS
        assert result.confidence == 0.0

    def test_whitespace_only(self, router: IntentRouter) -> None:
        result = router.classify("   ")
        assert result.intent in VALID_INTENTS
        assert result.confidence == 0.0

    def test_none_like_empty(self, router: IntentRouter) -> None:
        """Boş mesaj varsayılan intent döndürmeli."""
        result = router.classify("")
        assert result.intent == DEFAULT_INTENT


# ------------------------------------------------------------------
# Yapılandırma olmadan çalışma
# ------------------------------------------------------------------


class TestEmptyConfig:
    def test_empty_categories(self) -> None:
        config = IntentRouterConfig(categories={})
        router = IntentRouter(config)
        result = router.classify("merhaba")
        assert result.intent == DEFAULT_INTENT
        assert result.confidence == 0.0

    def test_category_with_no_utterances(self) -> None:
        config = IntentRouterConfig(
            categories={
                "chitchat": IntentCategory(
                    utterances=[],
                    action="direct_response",
                    response="Hi",
                )
            }
        )
        router = IntentRouter(config)
        result = router.classify("merhaba")
        assert result.intent == DEFAULT_INTENT
