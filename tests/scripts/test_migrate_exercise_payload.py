from scripts.migrate_exercise_payload import flat_to_payload, migrate_entry, needs_migration


def _flat_true_false():
    return {
        "exercise_id": "ex1",
        "concept_idx": 0,
        "concept_name": "C",
        "bloom_level": 1,
        "question": "Đúng hay sai?",
        "exercise_type": "true_false",
        "statement": "S",
        "correct_answer": True,
        "options": {},
        "explanation": "",
        "explanation_correct": "ok",
        "explanation_incorrect": "no",
        "timestamp": 1.0,
    }


def test_flat_to_payload_true_false():
    payload = flat_to_payload(_flat_true_false())
    assert payload == {"exercise_type": "true_false", "statement": "S", "correct_answer": True}


def test_flat_to_payload_ordering_uses_correct_answer_as_canonical():
    flat = {
        "exercise_type": "ordering",
        "items": ["x", "y", "z"],
        "correct_answer": ["a", "b", "c"],
    }
    assert flat_to_payload(flat) == {"exercise_type": "ordering", "correct_order": ["a", "b", "c"]}


def test_flat_to_payload_matching_rebuilds_pairs_from_correct_answer():
    flat = {
        "exercise_type": "matching",
        "correct_answer": {"L1": "R1", "L2": "R2"},
        "right_items": ["R2", "R1"],
    }
    assert flat_to_payload(flat) == {
        "exercise_type": "matching",
        "pairs": [{"left": "L1", "right": "R1"}, {"left": "L2", "right": "R2"}],
    }


def test_migrate_entry_strips_flat_keys_and_adds_payload():
    out = migrate_entry(_flat_true_false())
    assert out["payload"] == {
        "exercise_type": "true_false",
        "statement": "S",
        "correct_answer": True,
    }
    for flat in ("statement", "correct_answer", "options", "exercise_type"):
        assert flat not in out
    # envelope fields preserved
    assert out["question"] == "Đúng hay sai?"
    assert out["explanation_correct"] == "ok"


def test_needs_migration_is_idempotent():
    assert needs_migration(_flat_true_false()) is True
    assert needs_migration(migrate_entry(_flat_true_false())) is False
