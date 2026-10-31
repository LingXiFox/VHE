import pytest
from tools.new_experiment import create_record


def test_id_is_immutable(tmp_path):
    (tmp_path / "experiments").mkdir()
    loc = create_record(tmp_path, "EXP-0001", "first", "test")
    assert (loc / "metadata.json").is_file()
    assert (loc / "README.md").is_file()
    with pytest.raises(FileExistsError):
        create_record(tmp_path, "EXP-0001", "second", "should fail")


def test_bad_id(tmp_path):
    with pytest.raises(ValueError):
        create_record(tmp_path, "bad", "bad", "bad")
