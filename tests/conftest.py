import pytest

from core.config import load_config
from databases.create_student_db import create_database


@pytest.fixture
def config():
    config = load_config()
    config["model"]["backend"] = "mock"
    config["sampling"]["mode"] = "adaptive_sc"
    return config


@pytest.fixture
def db(tmp_path):
    return create_database(tmp_path / "students.sqlite")
