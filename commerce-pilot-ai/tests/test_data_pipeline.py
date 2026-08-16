from pathlib import Path

import pytest

from src.data_pipeline.download import ConfigurationError, load_config, planned_download, select_dataset
from src.data_pipeline.validate_raw_data import discover_files


def write_config(path: Path, raw_path: Path) -> None:
    path.write_text(
        "datasets:\n"
        "  olist:\n"
        "    source_url: https://example.invalid/olist.zip\n"
        f"    local_raw_data_path: '{raw_path.as_posix()}'\n"
        "    expected_file_format: zip\n",
        encoding="utf-8",
    )


def test_configuration_loading(tmp_path: Path) -> None:
    config_path = tmp_path / "datasets.yaml"
    write_config(config_path, tmp_path / "raw")
    assert load_config(config_path)["datasets"]["olist"]["expected_file_format"] == "zip"


def test_dataset_selection_rejects_unknown_name(tmp_path: Path) -> None:
    config_path = tmp_path / "datasets.yaml"
    write_config(config_path, tmp_path / "raw")
    with pytest.raises(ConfigurationError, match="Unknown dataset"):
        select_dataset(load_config(config_path), "combined")


def test_dry_run_planning_does_not_create_files(tmp_path: Path) -> None:
    config_path = tmp_path / "datasets.yaml"
    raw_path = tmp_path / "raw"
    write_config(config_path, raw_path)
    url, destination = planned_download(config_path, "olist")
    assert url == "https://example.invalid/olist.zip"
    assert destination.name == "olist.zip"
    assert not raw_path.exists()


def test_missing_file_validation_is_clear(tmp_path: Path) -> None:
    config_path = tmp_path / "datasets.yaml"
    raw_path = tmp_path / "raw"
    write_config(config_path, raw_path)
    with pytest.raises(FileNotFoundError, match="Raw-data directory not found"):
        discover_files(config_path, "olist")

