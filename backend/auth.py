"""Authentication helpers for user login, JWT creation, and bearer-token validation."""

from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .config import settings
from .database import SessionLocal, User

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
http_bearer = HTTPBearer(auto_error=False)


def authenticate_user(db: Session, username: str, password: str):
    """Validate username/password and return the user object if credentials are correct."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not pwd_context.verify(password, user.password_hash):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta):
    """Create a signed JWT token with an expiry claim."""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
):
    """Resolve and validate the current user from a bearer token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not credentials or credentials.scheme.lower() != "bearer":
        raise credentials_exception
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception


def init_user_db():
    """Create a default admin user when the user table is empty."""
    db = SessionLocal()
    try:
        if not db.query(User).filter_by(username="admin").first():
            hashed = pwd_context.hash("admin")
            db.add(User(username="admin", password_hash=hashed))
            db.commit()
    finally:
        db.close()


def get_user_by_username(db: Session, username: str):
    """Fetch a user by username."""
    return db.query(User).filter(User.username == username).first()


def create_user(db: Session, username: str, password: str):
    """Create and persist a new user with hashed password."""
    user = User(username=username, password_hash=pwd_context.hash(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
