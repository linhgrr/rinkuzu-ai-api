"""Shared infrastructure used across domains.

Keep this package import light so submodule imports like
``from api.shared import mongo_store`` do not eagerly pull the optional
LLM stack into unrelated tests.
"""
