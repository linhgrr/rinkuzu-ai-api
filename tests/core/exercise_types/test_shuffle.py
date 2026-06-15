from api.core.learning.exercise_types.shuffle import deterministic_shuffle


def test_same_seed_same_order():
    items = ["a", "b", "c", "d", "e"]
    assert deterministic_shuffle(items, "ex-1") == deterministic_shuffle(items, "ex-1")


def test_is_a_permutation_not_a_mutation():
    items = ["a", "b", "c", "d"]
    out = deterministic_shuffle(items, "ex-1")
    assert sorted(out) == sorted(items)
    assert items == ["a", "b", "c", "d"]  # input untouched


def test_different_seeds_generally_differ():
    items = [str(i) for i in range(10)]
    assert deterministic_shuffle(items, "ex-1") != deterministic_shuffle(items, "ex-2")
