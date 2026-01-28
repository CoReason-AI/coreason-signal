from typing import Any, Dict, List

from pydantic import BaseModel, SecretStr


class UserContext(BaseModel):
    user_id: SecretStr
    roles: List[str]
    metadata: Dict[str, Any]
