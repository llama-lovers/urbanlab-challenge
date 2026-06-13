import jwt
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import settings


def _user_or_ip(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        try:
            payload = jwt.decode(
                auth[7:], settings.jwt_secret, algorithms=[settings.jwt_algorithm]
            )
            return f"user:{payload['sub']}"
        except Exception:
            pass
    return get_remote_address(request)


limiter = Limiter(key_func=_user_or_ip)
