from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, LargeBinary, Float, DateTime
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    # 192-dimensional ECAPA-TDNN embedding stored as raw float32 bytes
    embedding = Column(LargeBinary, nullable=False)
    registration_score = Column(Float, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
