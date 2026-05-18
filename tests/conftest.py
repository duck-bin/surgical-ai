"""Shared pytest configuration.

The `gpu` marker is declared in pyproject.toml; CPU-only CI runs with
`-m "not gpu"`. Stub tests for not-yet-implemented steps are skipped.
"""
