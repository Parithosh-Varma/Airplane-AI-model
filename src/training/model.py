import logging
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

from config import TRAINING_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)


class AircraftModel(nn.Module):
    def __init__(self, num_classes: int, model_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.backbone = self._build_backbone(model_name, pretrained)
        self.feature_dim = self._get_feature_dim()
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def _build_backbone(self, name: str, pretrained: bool):
        if name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            return models.resnet50(weights=weights)
        elif name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            return models.vit_b_16(weights=weights)
        elif name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            return models.efficientnet_b3(weights=weights)
        else:
            logger.warning("Unknown model %s, falling back to resnet50", name)
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            return models.resnet50(weights=weights)

    def _get_feature_dim(self) -> int:
        if "resnet" in self.model_name:
            return self.backbone.fc.in_features
        elif "vit" in self.model_name:
            return self.backbone.heads.head.in_features
        elif "efficientnet" in self.model_name:
            return self.backbone.classifier[1].in_features
        return 2048

    def forward(self, x):
        if "resnet" in self.model_name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        elif "vit" in self.model_name:
            x = self.backbone._process_input(x)
            n = x.shape[0]
            cls_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = self.backbone.encoder(x)
            x = x[:, 0]
        elif "efficientnet" in self.model_name:
            x = self.backbone.features(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        else:
            x = self.backbone(x)
        return self.classifier(x)

    def save(self, path: Path = None):
        if path is None:
            path = MODELS_DIR / f"{self.model_name}_latest.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "model_name": self.model_name,
        }, str(path))
        logger.info("Model saved to %s", path)

    def load(self, path: Path):
        if not path.exists():
            logger.warning("No saved model at %s, starting from pretrained weights", path)
            return
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Model loaded from %s", path)
