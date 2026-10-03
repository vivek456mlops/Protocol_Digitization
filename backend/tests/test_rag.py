from fastapi.testclient import TestClient

from app.main import app, domain_context, domain_provenance, mock_data, retriever, store
from app.rag import Chunk, InMemoryRetriever, mock_answer


def test_api_root_is_available() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Protocol Atlas API"


def test_synthetic_dataset_contains_three_protocols() -> None:
    assert mock_data["synthetic"] is True
    assert len(mock_data["protocols"]) == 3
    assert len(retriever.chunks) >= 10


def test_retriever_prioritizes_matching_terms() -> None:
    retriever = InMemoryRetriever([
        Chunk("A", 1, "Incubate for five minutes."),
        Chunk("A", 2, "Centrifuge at 12000 g for two minutes."),
    ])
    results = retriever.search("What are the centrifuge settings?")
    assert results[0].page == 2


def test_mock_answer_is_grounded_in_sources() -> None:
    source = Chunk("A", 1, "Use 500 microliters of buffer.")
    assert "500 microliters" in mock_answer("buffer", [source])


def test_empty_retrieval_is_explicit() -> None:
    assert "could not find" in mock_answer("unknown", [])


def test_query_does_not_invent_answer_without_evidence() -> None:
    client = TestClient(app)
    response = client.post("/query", json={"question": "What is the moon phase?", "protocol": "Sample Preparation Protocol"})
    assert response.status_code == 200
    assert response.json()["answer"] is None
    assert response.json()["sources"] == []


def test_short_greeting_uses_conversational_path() -> None:
    client = TestClient(app)
    response = client.post("/query", json={"question": "hi", "protocol": "Sample Preparation Protocol"})
    assert response.status_code == 200
    assert response.json()["mode"] == "conversation"


def test_stretched_greeting_uses_conversational_path() -> None:
    client = TestClient(app)
    response = client.post("/query", json={"question": "hii", "protocol": "Sample Preparation Protocol"})
    assert response.status_code == 200
    assert response.json()["mode"] == "conversation"


def test_protocol_catalog_request_lists_available_names() -> None:
    client = TestClient(app)
    response = client.post("/query", json={"question": "Can you list all available protocol names?"})
    assert response.status_code == 200
    assert response.json()["mode"] == "catalog"
    assert "Sample Preparation Protocol" in response.json()["answer"]


def test_protocol_detail_request_uses_catalog_metadata() -> None:
    client = TestClient(app)
    response = client.post("/query", json={"question": "Tell me details about Sample Preparation Protocol."})
    assert response.status_code == 200
    assert response.json()["mode"] == "catalog"
    assert "8 pages" in response.json()["answer"]
    assert response.json()["sources"]


def test_natural_greeting_with_question_is_conversational() -> None:
    response = TestClient(app).post("/query", json={"question": "Hiii, how are you?"})
    assert response.status_code == 200
    assert response.json()["mode"] == "conversation"


def test_typo_tolerant_catalog_request_lists_protocols() -> None:
    response = TestClient(app).post("/query", json={"question": "lis of available prtocol"})
    assert response.status_code == 200
    assert response.json()["mode"] == "catalog"


def test_liters_question_converts_source_quantity() -> None:
    response = TestClient(app).post("/query", json={"question": "how many liters need to add for protocol Sample Preparation Protocol with sample tube"})
    assert response.status_code == 200
    assert "0.0005 liters" in response.json()["answer"]


def test_domain_context_supports_scientific_explanation() -> None:
    notes = domain_context("Why wash the column twice with wash solution?")
    assert notes
    assert "contaminant" in notes[0]


def test_domain_context_supports_protocol_deviation_questions() -> None:
    notes = domain_context("What happens if we increase the wash volume?")
    assert notes
    assert "protocol deviation" in notes[0]


def test_deviation_prompt_contains_no_recommendation_language() -> None:
    from app.providers import OpenAICompatibleAnswerProvider

    provider = OpenAICompatibleAnswerProvider("test", "https://example.invalid/v1", "test-model")
    assert "do not say the change might be beneficial" in provider._system_instruction()


def test_domain_context_has_provenance() -> None:
    provenance = domain_provenance("Why wash the column twice?")
    assert provenance[0]["kind"] == "curated"
    assert provenance[0]["source"]


def test_knowledge_sources_endpoint_is_available() -> None:
    response = TestClient(app).get("/knowledge/sources")
    assert response.status_code == 200
    assert response.json()["sources"]


def test_source_page_can_be_viewed() -> None:
    client = TestClient(app)
    response = client.get("/protocols/Sample%20Preparation%20Protocol/pages/3")
    assert response.status_code == 200
    assert "lysis buffer" in response.json()["text"]


def test_json_protocol_upload_is_indexed() -> None:
    client = TestClient(app)
    try:
        response = client.post(
            "/protocols/upload",
            files={"file": ("additional.json", b'{"protocols":[{"name":"Test Protocol","pages":1,"chunks":[{"page":1,"text":"Use the test buffer."}]}]}', "application/json")},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "indexed"
        assert any(chunk.protocol == "Test Protocol" for chunk in retriever.chunks)
    finally:
        store.remove(["Test Protocol"])
        retriever.chunks = store.chunks()
