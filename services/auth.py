"""JWT auth + multi-tenant helpers."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional

from flask import request, g, jsonify
from jose import jwt, JWTError
from passlib.context import CryptContext

from config import Config
from models import db, User, Tenant

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_token(user: User) -> str:
    payload = {
        "sub": user.id,
        "email": user.email,
        "tenant_id": user.tenant_id,
        "role": user.role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=Config.JWT_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, Config.JWT_SECRET, algorithm=Config.JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, Config.JWT_SECRET, algorithms=[Config.JWT_ALGORITHM])
    except JWTError:
        return None


def get_current_user() -> Optional[User]:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        data = decode_token(token)
        if data:
            return User.query.get(data.get("sub"))
    return None


def require_auth(optional: bool = False):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if user is None and not optional:
                return jsonify({"error": "unauthorized"}), 401
            g.user = user
            g.tenant_id = (
                user.tenant_id
                if user
                else request.headers.get("X-Tenant-Id")
                or request.args.get("tenant")
                or Config.DEFAULT_TENANT
            )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def ensure_default_tenant():
    t = Tenant.query.get(Config.DEFAULT_TENANT)
    if not t:
        t = Tenant(id=Config.DEFAULT_TENANT, name="Campus")
        db.session.add(t)
        db.session.commit()
    return t
