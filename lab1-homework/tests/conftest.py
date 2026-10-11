import pytest
from src.inference import load_classifier, load_encoder


@pytest.fixture(scope="session")
def encoder():
    return load_encoder()


@pytest.fixture(scope="session")
def classifier():
    return load_classifier()
