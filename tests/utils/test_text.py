import utils.text as text_utils


def test_clean_text_falls_back_without_underthesea(monkeypatch):
    monkeypatch.setattr(text_utils, "_text_normalize", None)

    cleaned = text_utils.clean_text("  Đại số!!!\n@@  ")

    assert cleaned == "Đại số!!!"
