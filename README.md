# Aviation ML Pipeline

Continuous 24/7 pipeline that scrapes aircraft photos from aviation sites, stores them, and periodically trains an AI model.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  JetPhotos      │────▶│              │────▶│  Image           │
│  Scraper        │     │  Database    │     │  Preprocessor    │
├─────────────────┤     │  (SQLite/    │     │  (OpenCV/PIL)    │
│  Airplane-      │────▶│   PostgreSQL)│────▶│                  │
│  Pictures       │     │              │     └────────┬─────────┘
│  Scraper        │     │              │              │
├─────────────────┤     │              │     ┌────────▼─────────┐
│  Planespotters  │────▶│              │     │  Model Trainer   │
│  Scraper        │     │              │     │  (PyTorch)       │
└─────────────────┘     └──────────────┘     │                  │
                                             └──────────────────┘
```

## Quick Start (Local)

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure (optional — defaults work out of the box)
cp .env.example .env

# 3. Run
python main.py
```

## Quick Start (Docker — Production)

```bash
# Copy and edit environment variables
cp .env.example .env

# Start the full stack (PostgreSQL + scraper)
docker compose up --build -d

# View logs
docker compose logs -f
```

## Project Structure

```
aviation-ml-pipeline/
├── main.py                           # Orchestrator — runs everything concurrently
├── config.py                         # Central config (env vars, paths, defaults)
├── src/
│   ├── scraper/
│   │   ├── base.py                   # Abstract scraper with retry/logic/delay/download
│   │   ├── jetphotos.py              # JetPhotos implementation
│   │   ├── airplanepictures.py       # Airplane-Pictures implementation
│   │   ├── planespotters.py          # Planespotters implementation
│   │   └── proxy_rotator.py          # Rotating proxy integration point
│   ├── database/
│   │   ├── schema.py                 # SQLAlchemy ORM: AircraftImage table
│   │   └── manager.py                # DB operations (dedup, insert, mark_trained, etc.)
│   ├── preprocessing/
│   │   └── image_processor.py        # Resize/normalize → ready_for_training/
│   └── training/
│       ├── dataset.py                # PyTorch Dataset (supports .npy + PIL)
│       ├── model.py                  # ResNet50/ViT/EfficientNet with classifier head
│       └── trainer.py                # Trigger-based training loop
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## How It Works

### Phase 1 — Scraping
- Three scrapers run as parallel async tasks, each targeting the "Latest Uploads" feed.
- Before downloading, the image URL is checked against the DB to prevent duplicates.
- Random 30–120s delays between requests + automatic proxy rotation support.
- Exponential backoff on 429/403/timeout (up to 5 retries).

### Phase 2 — Preprocessing
- A background worker picks up newly downloaded images, resizes to 224×224, normalizes, saves as `.npy` files in `data/ready_for_training/`.
- Marks images `is_preprocessed = True` in the DB.

### Phase 3 — Training
- A separate background worker checks the DB every 300s for untrained images.
- When count >= 500 (`MIN_NEW_IMAGES`), it triggers a fine-tuning session on a separate thread.
- Uses transfer learning (ResNet50 by default) with CosineAnnealingLR scheduler.
- Saves updated weights to `data/models/` and marks images `is_trained = True`.

### Phase 4 — 24/7 Operation
- All services run as asyncio tasks inside a single process with graceful signal handling.
- Docker Compose bundles PostgreSQL + the app for one-command deployment.

## Ethics & Compliance

- **robots.txt**: Each scraper respects rate limits; the delay range (30–120s) is globally configurable via `POLITENESS_DELAY_MIN/MAX`.
- **Proxies**: For 24/7 production use, you MUST configure residential rotating proxies. Recommended services:
  - [BrightData](https://brightdata.com/) (residential proxy network)
  - [Webshare](https://www.webshare.io/) (affordable rotating proxies)
  - [Smartproxy](https://smartproxy.com/)
- **Usage**: This is intended for educational/research purposes. Check each site's Terms of Service before deploying.

## Configuration

All config via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `POLITENESS_DELAY_MIN` | 30 | Min seconds between requests |
| `POLITENESS_DELAY_MAX` | 120 | Max seconds between requests |
| `MAX_RETRIES` | 5 | Retries per failed request |
| `MIN_NEW_IMAGES` | 500 | Images needed to trigger training |
| `MODEL_NAME` | resnet50 | resnet50 / vit_b_16 / efficientnet_b3 |
| `TORCH_DEVICE` | cpu | cpu or cuda |

## Monitoring

Logs are written to `logs/pipeline.log` and stdout. Stats are logged every 10 minutes:
```
STATS: {'total_images': 1200, 'untrained': 0, 'preprocessed': 1200}
```

For production, integrate with your logging infrastructure (e.g., `journald`, CloudWatch, Loki).

## Railway Deployment

### One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template)

### Manual Setup

1. **Install the Railway CLI**
   ```bash
   curl -fsSL https://railway.app/install.sh | sh
   ```

2. **Login and link your project**
   ```bash
   railway login
   railway init
   ```

3. **Add a PostgreSQL plugin**
   ```bash
   # Via Railway Dashboard:
   # New → PostgreSQL → Add
   #
   # Railway automatically injects DATABASE_URL into your service.
   ```

4. **Add a Volume for persistent data**
   ```bash
   # Via Railway Dashboard:
   # Select your service → Volumes → Add Volume
   # Mount path: /app/data
   ```
   This persists downloaded images and trained models across deploys.

5. **Set environment variables**
   ```bash
   railway variables --set POLITENESS_DELAY_MIN=30
   railway variables --set POLITENESS_DELAY_MAX=120
   railway variables --set MIN_NEW_IMAGES=500
   railway variables --set MODEL_NAME=resnet50
   ```

6. **Deploy**
   ```bash
   railway up
   # or trigger a deploy from the Railway Dashboard
   ```

### What Railway Provides

| Requirement | Railway Solution |
|---|---|
| **PostgreSQL** | Built-in plugin → auto-injects `DATABASE_URL` |
| **Docker build** | Automatic via `railway.json` (DOCKERFILE builder) |
| **Health checks** | `/health` endpoint on `PORT` (port 8080) |
| **Auto-restart** | Railway restarts on crash (max 10 retries) |
| **Persistent data** | Railway Volumes mounted at `/app/data` |
| **Logs** | Railway Dashboard + `railway logs` |
| **Scaling** | `numReplicas` in `railway.json` |

### Railway-Specific Config

The pipeline auto-detects Railway:

- `DATABASE_URL` from Railway's PostgreSQL plugin is converted from `postgresql://...` to `postgresql+asyncpg://...` automatically
- A lightweight healthcheck server starts on `$PORT` (usually 8080) when the `PORT` env var is present
- The file logger degrades gracefully if `logs/` is not writable (logs always go to stdout)

### Environment Variables for Railway

Set these in the Railway Dashboard or via the CLI:

```
USE_PLAYWRIGHT=true
POLITENESS_DELAY_MIN=60
POLITENESS_DELAY_MAX=180
MAX_RETRIES=3
MIN_NEW_IMAGES=500
MODEL_NAME=resnet50
TORCH_DEVICE=cpu
LOG_LEVEL=INFO
```

> **Note:** Railway provides `DATABASE_URL` and `PORT` automatically — do **not** set them manually.
>
> **Tip:** Set `USE_PLAYWRIGHT=true` to use headless Chromium browsers instead of plain HTTP requests. This bypasses most anti-bot protections because the sites see a real browser. Playwright Chromium (~300MB) is pre-installed in the Docker image.
