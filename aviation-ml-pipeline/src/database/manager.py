import logging
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from src.database.schema import Base, AircraftImage, init_db

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, database_url: str, echo: bool = False):
        self.database_url = database_url
        self.engine, self.SessionLocal = init_db(database_url, echo=echo)
        logger.info("Database initialized at %s", database_url)

    def get_session(self) -> Session:
        return self.SessionLocal()

    def image_exists(self, image_url: str) -> bool:
        with self.get_session() as session:
            return session.query(
                session.query(AircraftImage).filter(
                    AircraftImage.image_url == image_url
                ).exists()
            ).scalar()

    def insert_image(self, image_data: dict) -> Optional[int]:
        with self.get_session() as session:
            existing = session.query(AircraftImage).filter(
                AircraftImage.image_url == image_data["image_url"]
            ).first()
            if existing:
                return existing.id
            record = AircraftImage(**image_data)
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id

    def bulk_insert_images(self, records: list[dict]) -> list[int]:
        ids = []
        with self.get_session() as session:
            for data in records:
                existing = session.query(AircraftImage).filter(
                    AircraftImage.image_url == data["image_url"]
                ).first()
                if existing:
                    ids.append(existing.id)
                    continue
                record = AircraftImage(**data)
                session.add(record)
                session.flush()
                ids.append(record.id)
            session.commit()
        return ids

    def mark_trained(self, image_ids: list[int]):
        with self.get_session() as session:
            session.query(AircraftImage).filter(
                AircraftImage.id.in_(image_ids)
            ).update(
                {AircraftImage.is_trained: True, AircraftImage.updated_at: datetime.utcnow()},
                synchronize_session=False,
            )
            session.commit()

    def mark_preprocessed(self, image_ids: list[int]):
        with self.get_session() as session:
            session.query(AircraftImage).filter(
                AircraftImage.id.in_(image_ids)
            ).update(
                {
                    AircraftImage.is_preprocessed: True,
                    AircraftImage.updated_at: datetime.utcnow(),
                },
                synchronize_session=False,
            )
            session.commit()

    def update_filepath(self, image_id: int, filepath: str):
        with self.get_session() as session:
            session.query(AircraftImage).filter(
                AircraftImage.id == image_id
            ).update(
                {
                    AircraftImage.local_filepath: filepath,
                    AircraftImage.updated_at: datetime.utcnow(),
                }
            )
            session.commit()

    def count_untrained(self) -> int:
        with self.get_session() as session:
            return session.query(func.count(AircraftImage.id)).filter(
                AircraftImage.is_trained == False,
                AircraftImage.is_preprocessed == True,
            ).scalar()

    def get_untrained_batch(self, limit: int = 500) -> List[AircraftImage]:
        with self.get_session() as session:
            return session.query(AircraftImage).filter(
                AircraftImage.is_trained == False,
                AircraftImage.is_preprocessed == True,
            ).limit(limit).all()

    def get_unprocessed_images(self, limit: int = 100) -> List[AircraftImage]:
        with self.get_session() as session:
            return session.query(AircraftImage).filter(
                AircraftImage.is_preprocessed == False,
                AircraftImage.local_filepath.isnot(None),
            ).limit(limit).all()

    def get_stats(self) -> dict:
        with self.get_session() as session:
            total = session.query(func.count(AircraftImage.id)).scalar()
            untrained = session.query(func.count(AircraftImage.id)).filter(
                AircraftImage.is_trained == False
            ).scalar()
            preprocessed = session.query(func.count(AircraftImage.id)).filter(
                AircraftImage.is_preprocessed == True
            ).scalar()
            return {
                "total_images": total,
                "untrained": untrained,
                "preprocessed": preprocessed,
            }
