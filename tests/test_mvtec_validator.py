from pathlib import Path

import pytest

from metinspect.data.mvtec import validate_mvtec_root


def test_validate_mvtec_root_missing(tmp_path: Path):
    missing = tmp_path / "mvtec_ad"
    with pytest.raises(FileNotFoundError):
        validate_mvtec_root(missing)
