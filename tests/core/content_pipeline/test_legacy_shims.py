from api.core.content_pipeline.infrastructure.llm import get_llm as infra_get_llm
from api.core.content_pipeline.infrastructure.embed import (
    EmbeddingClient as infra_embedding_client,
)
from api.core.content_pipeline.infrastructure.merge import (
    merge_by_name as infra_merge_by_name,
)
from api.core.content_pipeline.infrastructure.prompts import (
    CYCLE_REMOVAL_PROMPT as infra_cycle_removal_prompt,
)
from api.core.content_pipeline.infrastructure.utils import clean_text as infra_clean_text
from embed import EmbeddingClient as legacy_embedding_client
from llm import get_llm as legacy_get_llm
from merge import merge_by_name as legacy_merge_by_name
from prompts import CYCLE_REMOVAL_PROMPT as legacy_cycle_removal_prompt
from utils import clean_text as legacy_clean_text


def test_legacy_root_shims_point_to_infrastructure_modules():
    assert legacy_get_llm is infra_get_llm
    assert legacy_embedding_client is infra_embedding_client
    assert legacy_merge_by_name is infra_merge_by_name
    assert legacy_clean_text is infra_clean_text
    assert legacy_cycle_removal_prompt == infra_cycle_removal_prompt
