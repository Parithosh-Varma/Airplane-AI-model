from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, create_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker


class Base(DeclarativeBase):
    pass


class AircraftImage(Base):
    __tablename__ = "aircraft_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_url = Column(String(1024), nullable=False, unique=True)
    image_url = Column(String(1024), nullable=False)
    local_filepath = Column(String(1024), nullable=True)
    aircraft_name = Column(String(512), nullable=True)
    description = Column(Text, nullable=True)
    source_site = Column(String(128), nullable=False)
    download_timestamp = Column(DateTime, default=datetime.utcnow)
    is_trained = Column(Boolean, default=False)
    is_preprocessed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AircraftImage(id={self.id}, source='{self.source_site}', trained={self.is_trained})>"


def init_db(database_url: str, echo: bool = False):
    engine = create_engine(database_url, echo=echo)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return engine, SessionLocal
