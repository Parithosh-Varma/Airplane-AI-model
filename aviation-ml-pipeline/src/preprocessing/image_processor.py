import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import PREPROCESSING_CONFIG, DATA_DIR
from src.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.target_size = tuple(PREPROCESSING_CONFIG["target_size"])
        self.normalize_mean = np.array(PREPROCESSING_CONFIG["normalize_mean"])
        self.normalize_std = np.array(PREPROCESSING_CONFIG["normalize_std"])
        self.ready_dir = DATA_DIR / "ready_for_training"
        self.ready_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_single(self, input_path: str) -> tuple[bool, str]:
        input_path = Path(input_path)
        if not input_path.exists():
            logger.warning("File not found: %s", input_path)
            return False, ""

        try:
            img = cv2.imread(str(input_path))
            if img is None:
                img_pil = Image.open(input_path).convert("RGB")
                img = np.array(img_pil)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            img = img.astype(np.float32) / 255.0
            img = (img - self.normalize_mean) / (self.normalize_std + 1e-8)
            img = img.transpose(2, 0, 1)
            img = img.astype(np.float32)

            output_filename = f"{input_path.stem}.npy"
            output_path = self.ready_dir / output_filename
            np.save(str(output_path), img)
            return True, str(output_path)

        except Exception as e:
            logger.error("Failed to preprocess %s: %s", input_path, e)
            return False, ""

    async def preprocessing_loop(self):
        while True:
            try:
                images = self.db.get_unprocessed_images(limit=50)
                if not images:
                    await asyncio.sleep(30)
                    continue

                processed_ids = []
                for img_record in images:
                    if not img_record.local_filepath:
                        continue
                    success, output_path = self.preprocess_single(img_record.local_filepath)
                    if success:
                        self.db.update_filepath(img_record.id, output_path)
                        processed_ids.append(img_record.id)

                if processed_ids:
                    self.db.mark_preprocessed(processed_ids)
                    logger.info("Preprocessed %d images", len(processed_ids))

                await asyncio.sleep(5)
            except Exception as e:
                logger.exception("Preprocessing loop error: %s", e)
                await asyncio.sleep(30)
