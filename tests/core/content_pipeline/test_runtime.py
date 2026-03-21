from pathlib import Path

from api.core.content_pipeline.infrastructure.runtime import calculate_file_hash


def test_calculate_file_hash_is_stable_for_same_content(tmp_path: Path):
    file_path = tmp_path / "lesson.txt"
    file_path.write_text("algebra", encoding="utf-8")

    first_hash = calculate_file_hash(str(file_path))
    second_hash = calculate_file_hash(str(file_path))

    assert first_hash == second_hash
    assert len(first_hash) == 64
