import os
import sys

# Ensure local mock libs are available for tests if real package is missing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "libs")))

import pytest
from coreason_identity.models import UserContext
from coreason_identity.types import SecretStr


@pytest.fixture  # type: ignore[misc]
def user_context() -> UserContext:
    return UserContext(user_id=SecretStr("test-user"), roles=["tester"], metadata={"env": "test"})
