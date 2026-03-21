from api.core.content_pipeline.infrastructure.llm import get_llm as infra_get_llm
from api.core.content_pipeline.infrastructure.embed import (
    EmbeddingClient as infra_embedding_client,
)
from api.core.content_pipeline.infrastructure.graph import (
    KnowledgeGraphBuilder as infra_graph_builder,
)
from api.core.content_pipeline.infrastructure.merge import (
    merge_by_name as infra_merge_by_name,
)
from api.core.content_pipeline.infrastructure.processors import (
    FileLoaderFactory as infra_file_loader_factory,
)
from api.core.content_pipeline.infrastructure.prompts import (
    CYCLE_REMOVAL_PROMPT as infra_cycle_removal_prompt,
)
from api.core.content_pipeline.infrastructure.utils import clean_text as infra_clean_text
from embed import EmbeddingClient as legacy_embedding_client
from graph import KnowledgeGraphBuilder as legacy_graph_builder
from llm import get_llm as legacy_get_llm
from merge import merge_by_name as legacy_merge_by_name
from processors import FileLoaderFactory as legacy_file_loader_factory
from prompts import CYCLE_REMOVAL_PROMPT as legacy_cycle_removal_prompt
from utils import clean_text as legacy_clean_text


def test_legacy_root_shims_point_to_infrastructure_modules():
    assert legacy_get_llm is infra_get_llm
    assert legacy_embedding_client is infra_embedding_client
    assert legacy_graph_builder is infra_graph_builder
    assert legacy_merge_by_name is infra_merge_by_name
    assert legacy_file_loader_factory is infra_file_loader_factory
    assert legacy_clean_text is infra_clean_text
    assert legacy_cycle_removal_prompt == infra_cycle_removal_prompt
