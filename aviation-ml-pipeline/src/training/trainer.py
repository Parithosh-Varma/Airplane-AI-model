import asyncio
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import TRAINING_CONFIG, MODELS_DIR
from src.database.manager import DatabaseManager
from src.training.dataset import AircraftImageDataset
from src.training.model import AircraftModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.device = torch.device(TRAINING_CONFIG["device"])
        logger.info("Using device: %s", self.device)

        self.model = AircraftModel(
            num_classes=TRAINING_CONFIG["num_classes"],
            model_name=TRAINING_CONFIG["model_name"],
            pretrained=TRAINING_CONFIG["pretrained"],
        ).to(self.device)

        latest_path = MODELS_DIR / f'{TRAINING_CONFIG["model_name"]}_latest.pt'
        if latest_path.exists():
            self.model.load(latest_path)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=1e-4,
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def train_on_batch(self, image_ids: list[int], image_paths: list[str], labels: list[int]):
        if not image_paths:
            logger.info("No images to train on")
            return

        dataset = AircraftImageDataset(image_paths, labels, augment=True)
        loader = DataLoader(
            dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        logger.info(
            "Starting training on %d images (%d batches)",
            len(dataset), len(loader),
        )

        for epoch in range(TRAINING_CONFIG["epochs_per_batch"]):
            loss = self.train_epoch(loader)
            logger.info("Epoch %d/%d: loss=%.4f", epoch + 1, TRAINING_CONFIG["epochs_per_batch"], loss)

        self.model.save()
        self.db.mark_trained(image_ids)
        logger.info("Training complete. %d images marked as trained.", len(image_ids))

    async def training_loop(self):
        while True:
            try:
                count = self.db.count_untrained()
                min_new = TRAINING_CONFIG["min_new_images"]

                if count >= min_new:
                    logger.info(
                        "Found %d untrained images (threshold: %d). Starting training...",
                        count, min_new,
                    )
                    batch = self.db.get_untrained_batch(limit=count)
                    if not batch:
                        await asyncio.sleep(TRAINING_CONFIG["check_interval_seconds"])
                        continue

                    image_ids = [r.id for r in batch]
                    image_paths = [r.local_filepath for r in batch if r.local_filepath]
                    labels = [hash(r.aircraft_name or "") % TRAINING_CONFIG["num_classes"] for r in batch]

                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None,
                        self.train_on_batch,
                        image_ids,
                        image_paths,
                        labels,
                    )
                else:
                    logger.debug(
                        "Untrained images: %d/%d. Sleeping...", count, min_new,
                    )

                await asyncio.sleep(TRAINING_CONFIG["check_interval_seconds"])
            except Exception as e:
                logger.exception("Training loop error: %s", e)
                await asyncio.sleep(60)
