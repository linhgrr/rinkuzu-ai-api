from api.core.content_pipeline.infrastructure.llm import get_llm as infra_get_llm
from api.core.content_pipeline.infrastructure.prompts import (
    CYCLE_REMOVAL_PROMPT as infra_cycle_removal_prompt,
)
from api.core.content_pipeline.infrastructure.utils import clean_text as infra_clean_text
from llm import get_llm as legacy_get_llm
from prompts import CYCLE_REMOVAL_PROMPT as legacy_cycle_removal_prompt
from utils import clean_text as legacy_clean_text


def test_legacy_root_shims_point_to_infrastructure_modules():
    assert legacy_get_llm is infra_get_llm
    assert legacy_clean_text is infra_clean_text
    assert legacy_cycle_removal_prompt == infra_cycle_removal_prompt
