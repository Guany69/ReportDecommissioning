import sys
from pathlib import Path

# Make the project root importable so `import report_cleanup` works under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from report_cleanup.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()
