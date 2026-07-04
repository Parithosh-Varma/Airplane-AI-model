import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_raw_db_url = os.getenv("DATABASE_URL", f"sqlite+pysqlite:///{DATA_DIR / 'pipeline.db'}")
if _raw_db_url.startswith("postgresql://"):
    if "+asyncpg" in _raw_db_url:
        _raw_db_url = _raw_db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    elif "+psycopg2" not in _raw_db_url and _raw_db_url.count("://") == 1:
        _raw_db_url = _raw_db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    os.environ["DATABASE_URL"] = _raw_db_url
DATABASE_URL = _raw_db_url

RAILWAY_PORT = int(os.getenv("PORT", os.getenv("RAILWAY_PORT", "0")))

SCRAPER_CONFIG = {
    "politeness_delay_min": int(os.getenv("POLITENESS_DELAY_MIN", "30")),
    "politeness_delay_max": int(os.getenv("POLITENESS_DELAY_MAX", "120")),
    "max_retries": int(os.getenv("MAX_RETRIES", "5")),
    "backoff_base": int(os.getenv("BACKOFF_BASE", "2")),
    "request_timeout": int(os.getenv("REQUEST_TIMEOUT", "30")),
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ],
    "proxy_rotation_url": os.getenv("PROXY_ROTATION_URL", ""),
    "proxy_api_key": os.getenv("PROXY_API_KEY", ""),
}

TRAINING_CONFIG = {
    "batch_size": int(os.getenv("TRAIN_BATCH_SIZE", "32")),
    "epochs_per_batch": int(os.getenv("EPOCHS_PER_BATCH", "5")),
    "learning_rate": float(os.getenv("LEARNING_RATE", "1e-4")),
    "min_new_images": int(os.getenv("MIN_NEW_IMAGES", "500")),
    "check_interval_seconds": int(os.getenv("CHECK_INTERVAL_SECONDS", "300")),
    "model_name": os.getenv("MODEL_NAME", "resnet50"),
    "image_size": (224, 224),
    "num_classes": int(os.getenv("NUM_CLASSES", "100")),
    "device": os.getenv("TORCH_DEVICE", "cpu"),
    "pretrained": True,
}

PREPROCESSING_CONFIG = {
    "target_size": (224, 224),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "output_format": "jpg",
    "quality": 90,
}

LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
}
