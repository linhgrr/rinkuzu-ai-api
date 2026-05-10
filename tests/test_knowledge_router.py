import pytest

from api.exceptions import SessionNotFoundError
from api.routers import knowledge as knowledge_router
from api.routers.knowledge import _get_session_resource
from api.schemas import ConceptDetailResponse, KnowledgeGraphResponse, MasteryMatrixResponse


@pytest.mark.anyio
async def test_get_session_resource_supports_sync_knowledge_graph_fetcher(monkeypatch):
    manager = object()

    async def fake_resolve_user_session(manager, session_id: str, user_id: str):
        assert manager is not None
        assert session_id == "session-1"
        assert user_id == "user-1"
        return None

    monkeypatch.setattr(knowledge_router, "resolve_user_session", fake_resolve_user_session)

    response = await _get_session_resource(
        manager,
        "session-1",
        "user-1",
        fetcher=lambda: {
            "nodes": [
                {
                    "id": "c1",
                    "index": 0,
                    "name": "Alpha",
                    "mastery": 0.5,
                    "status": "available",
                    "visited": True,
                }
            ],
            "edges": [],
        },
        response_cls=KnowledgeGraphResponse,
    )

    assert response == {
        "success": True,
        "data": {
            "nodes": [
                {
                    "id": "c1",
                    "index": 0,
                    "name": "Alpha",
                    "mastery": 0.5,
                    "status": "available",
                    "visited": True,
                }
            ],
            "edges": [],
        },
    }


@pytest.mark.anyio
async def test_get_session_resource_supports_sync_mastery_fetcher(monkeypatch):
    async def fake_resolve_user_session(manager, session_id: str, user_id: str):
        del manager, session_id, user_id
        return None

    monkeypatch.setattr(knowledge_router, "resolve_user_session", fake_resolve_user_session)

    response = await _get_session_resource(
        object(),
        "session-1",
        "user-1",
        fetcher=lambda: {
            "matrix": [
                {
                    "concept_id": "c1",
                    "concept_name": "Alpha",
                    "bloom_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                }
            ],
            "bloom_labels": ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
        },
        response_cls=MasteryMatrixResponse,
    )

    assert response["success"] is True
    assert response["data"]["matrix"][0]["concept_id"] == "c1"


@pytest.mark.anyio
async def test_get_session_resource_supports_sync_concept_detail_fetcher(monkeypatch):
    async def fake_resolve_user_session(manager, session_id: str, user_id: str):
        del manager, session_id, user_id
        return None

    monkeypatch.setattr(knowledge_router, "resolve_user_session", fake_resolve_user_session)

    response = await _get_session_resource(
        object(),
        "session-1",
        "user-1",
        fetcher=lambda: {
            "id": "c1",
            "name": "Alpha",
            "definition": "alpha def",
            "mastery": 0.5,
            "status": "available",
            "bloom_mastery": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "prerequisites": [],
            "dependents": [],
            "visited": True,
            "visit_count": 2,
        },
        response_cls=ConceptDetailResponse,
    )

    assert response["success"] is True
    assert response["data"]["id"] == "c1"


@pytest.mark.anyio
async def test_get_session_resource_raises_session_not_found_when_fetcher_returns_none(monkeypatch):
    async def fake_resolve_user_session(manager, session_id: str, user_id: str):
        del manager, session_id, user_id
        return None

    monkeypatch.setattr(knowledge_router, "resolve_user_session", fake_resolve_user_session)

    with pytest.raises(SessionNotFoundError):
        await _get_session_resource(
            object(),
            "session-1",
            "user-1",
            fetcher=lambda: None,
            response_cls=KnowledgeGraphResponse,
        )
