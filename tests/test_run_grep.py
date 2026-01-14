import os
import tempfile

import pytest
from furax_cs.r_analysis.run_grep import (
    expand_pattern_with_captures,
    is_regex_pattern,
    match_folder_with_regex_tokens,
    match_token_regex,
    run_grep,
)


class TestIsRegexPattern:
    def test_plain_token_not_regex(self):
        assert not is_regex_pattern("kmeans_BD200")

    def test_or_syntax_not_regex(self):
        assert not is_regex_pattern("kmeans_(GAL020|GAL040)")

    def test_digit_capture_is_regex(self):
        assert is_regex_pattern(r"kmeans_BD(\d+)")

    def test_word_capture_is_regex(self):
        assert is_regex_pattern(r"kmeans_(\w+)")

    def test_character_class_is_regex(self):
        assert is_regex_pattern(r"kmeans_([a-z]+)")

    def test_star_quantifier_is_regex(self):
        assert is_regex_pattern(r"kmeans_BD(\d*)")

    def test_plus_quantifier_is_regex(self):
        assert is_regex_pattern(r"kmeans_BD(\d+)")

    def test_no_parentheses_not_regex(self):
        assert not is_regex_pattern("kmeans_BD200_GAL020")


class TestExpandPatternWithCaptures:
    def test_single_capture(self):
        pattern_tokens = [r"BD(\d+)"]
        captures = {0: "BD200"}
        result = expand_pattern_with_captures(pattern_tokens, captures)
        assert result == "BD200"

    def test_multiple_captures(self):
        pattern_tokens = [r"BD(\d+)", r"GAL(\d+)"]
        captures = {0: "BD200", 1: "GAL020"}
        result = expand_pattern_with_captures(pattern_tokens, captures)
        assert result == "BD200_GAL020"

    def test_with_prefix(self):
        pattern_tokens = ["kmeans", r"BD(\d+)"]
        captures = {1: "BD2500"}
        result = expand_pattern_with_captures(pattern_tokens, captures)
        assert result == "kmeans_BD2500"

    def test_preserves_non_capture_text(self):
        pattern_tokens = ["prefix", r"(\d+)", "suffix"]
        captures = {1: "123"}
        result = expand_pattern_with_captures(pattern_tokens, captures)
        assert result == "prefix_123_suffix"


class TestMatchTokenRegex:
    def test_matches_digits(self):
        folder_tokens = ["kmeans", "BD200", "GAL020"]
        result = match_token_regex(folder_tokens, r"BD(\d+)")
        assert result == "BD200"

    def test_no_match_returns_none(self):
        folder_tokens = ["kmeans", "ABC", "GAL020"]
        result = match_token_regex(folder_tokens, r"BD(\d+)")
        assert result is None


class TestMatchFolderWithRegexTokens:
    def test_matches_simple_pattern(self):
        folder_tokens = ["kmeans", "c1d1s1", "BD200", "TD500", "GAL020"]
        pattern_tokens = ["kmeans", r"BD(\d+)"]
        matched, captures = match_folder_with_regex_tokens(folder_tokens, pattern_tokens)
        assert matched is True
        assert captures == {1: "BD200"}

    def test_matches_multiple_regex_tokens(self):
        folder_tokens = ["kmeans", "BD200", "GAL020"]
        pattern_tokens = [r"BD(\d+)", r"GAL(\d+)"]
        matched, captures = match_folder_with_regex_tokens(folder_tokens, pattern_tokens)
        assert matched is True
        assert captures == {0: "BD200", 1: "GAL020"}

    def test_no_match_returns_false(self):
        folder_tokens = ["kmeans", "ABC", "GAL020"]
        pattern_tokens = ["kmeans", r"BD(\d+)"]
        matched, captures = match_folder_with_regex_tokens(folder_tokens, pattern_tokens)
        assert matched is False
        assert captures == {}


class TestRunGrepRegex:
    @pytest.fixture
    def fake_results_dir(self):
        """Create fake folder structure for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            folders = [
                "kmeans_c1d1s1_BD200_TD500_LiteBIRD_GAL020",
                "kmeans_c1d1s1_BD200_TD500_LiteBIRD_GAL040",
                "kmeans_c1d1s1_BD2500_TD500_LiteBIRD_GAL020",
                "kmeans_c1d1s1_BD2500_TD500_LiteBIRD_GAL040",
                "kmeans_c1d1s1_BD3000_TD500_LiteBIRD_GAL020",
                "fgbuster_c1d1s1_BD10000_LiteBIRD_GAL020",
            ]
            for folder in folders:
                os.makedirs(os.path.join(tmpdir, folder))
            yield tmpdir

    def test_regex_groups_by_captured_value(self, fake_results_dir):
        r"""Pattern BD(\d+) should create separate groups for each BD value."""
        result = run_grep(fake_results_dir, [r"kmeans_BD(\d+)"])

        # Should have 3 groups: BD200, BD2500, BD3000
        assert "kmeans_BD200" in result
        assert "kmeans_BD2500" in result
        assert "kmeans_BD3000" in result
        assert len(result) == 3

        # BD200 should have 2 folders (GAL020 and GAL040)
        bd200_folders, _, _ = result["kmeans_BD200"]
        assert len(bd200_folders) == 2

    def test_plain_token_matching_still_works(self, fake_results_dir):
        """Non-regex patterns should use token matching."""
        result = run_grep(fake_results_dir, ["kmeans_BD200"])

        assert "kmeans_BD200" in result
        folders, _, _ = result["kmeans_BD200"]
        assert len(folders) == 2  # GAL020 and GAL040

    def test_multiple_capture_groups(self, fake_results_dir):
        """Multiple captures create flat expanded name."""
        result = run_grep(fake_results_dir, [r"kmeans_BD(\d+)_GAL(\d+)"])

        # Each unique BD+GAL combination is a separate group
        assert "kmeans_BD200_GAL020" in result
        assert "kmeans_BD200_GAL040" in result
        assert "kmeans_BD2500_GAL020" in result

    def test_fgbuster_pattern(self, fake_results_dir):
        """Test regex on fgbuster folders."""
        result = run_grep(fake_results_dir, [r"fgbuster_BD(\d+)"])

        assert "fgbuster_BD10000" in result
        folders, _, _ = result["fgbuster_BD10000"]
        assert len(folders) == 1

    def test_or_syntax_still_works(self, fake_results_dir):
        """OR syntax (GAL020|GAL040) should match either token."""
        result = run_grep(fake_results_dir, ["kmeans_(GAL020|GAL040)"])

        # Should match all kmeans folders with GAL020 or GAL040
        assert "kmeans_(GAL020|GAL040)" in result
        folders, _, _ = result["kmeans_(GAL020|GAL040)"]
        # All 5 kmeans folders have either GAL020 or GAL040
        assert len(folders) == 5

    def test_mixed_specs(self, fake_results_dir):
        """Test mix of regex and non-regex patterns."""
        result = run_grep(fake_results_dir, ["fgbuster", r"kmeans_BD(\d+)"])

        # fgbuster should be token match
        assert "fgbuster" in result
        fgb_folders, _, _ = result["fgbuster"]
        assert len(fgb_folders) == 1

        # kmeans_BD should expand to multiple groups
        assert "kmeans_BD200" in result
        assert "kmeans_BD2500" in result
        assert "kmeans_BD3000" in result

    def test_index_spec_preserved(self, fake_results_dir):
        """Index spec should be preserved through regex expansion."""
        result = run_grep(fake_results_dir, [r"kmeans_BD(\d+),0-2"])

        # Index spec should be (0, 2) for all expanded groups
        for key in result:
            _, index_spec, _ = result[key]
            assert index_spec == (0, 2)
