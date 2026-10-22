"""Tests for the extractive-compression noise filter."""

from __future__ import annotations

from compress import is_relevant


def test_relevant_text_is_kept():
    assert is_relevant("Chunking splits documents into overlapping pieces.")


def test_empty_extract_is_dropped():
    assert not is_relevant("")
    assert not is_relevant("   \n  ")


def test_noise_marker_is_dropped_case_insensitively():
    assert not is_relevant("NO RELEVANT INFORMATION")
    assert not is_relevant("no relevant information.")
