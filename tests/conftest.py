import pytest
from coreason_identity.models import UserContext
from coreason_identity.types import SecretStr

@pytest.fixture  # type: ignore[misc]
def user_context() -> UserContext:
    return UserContext(user_id=SecretStr("test-user"), roles=["tester"], metadata={"env": "test"})
