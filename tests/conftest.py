import pytest

from per3clock import data


@pytest.fixture(scope="session")
def meta():
    return data.load_metadata()


@pytest.fixture(scope="session")
def clock():
    return data.load_clock_genes()
